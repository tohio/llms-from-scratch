import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── RMS Normalisation ────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-8):
        super().__init__()
        self.eps   = eps

        # Learnable scale parameter — no bias, RMSNorm only scales
        self.scale = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        # Compute root mean square across the embedding dimension
        rms = x.pow(2).mean(dim=-1, keepdim=True).sqrt()

        # Normalise by RMS then apply learnable scale
        return self.scale * x / (rms + self.eps)


# ─── RoPE ─────────────────────────────────────────────────────────────────────

def get_rope_frequencies(head_dim, seq_len, base=10000):
    # Compute theta for each dimension pair: θ = 1 / (10000 ^ (2i / head_dim))
    # torch.arange(0, head_dim, 2) gives [0, 2, 4, ...] — one per pair
    thetas    = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))

    # Positions in the sequence [0, 1, 2, ..., seq_len-1]
    positions = torch.arange(seq_len).float()

    # Outer product — every position multiplied by every theta
    # shape: (seq_len, head_dim/2)
    return torch.outer(positions, thetas)


def apply_rope(x, freqs):
    # x shape: (b, num_heads, seq_len, head_dim)
    # freqs shape: (seq_len, head_dim/2)

    # Split x into pairs along the last dimension
    x1 = x[..., 0::2]  # even indices — first of each pair
    x2 = x[..., 1::2]  # odd indices  — second of each pair

    # Compute sin and cos of the rotation angles
    cos = freqs.cos().unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim/2)
    sin = freqs.sin().unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim/2)

    # Apply the rotation to each pair
    # [x1, x2] → [x1·cos(θ) - x2·sin(θ), x1·sin(θ) + x2·cos(θ)]
    rotated_x1 = x1 * cos - x2 * sin
    rotated_x2 = x1 * sin + x2 * cos

    # Interleave rotated pairs back into the original shape
    out = torch.stack([rotated_x1, rotated_x2], dim=-1)
    out = out.flatten(-2)

    return out


# ─── Grouped Query Attention with RoPE ───────────────────────────────────────

class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, num_heads, num_kv_heads):
        super().__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        assert num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"

        self.d_model      = d_model
        self.num_heads    = num_heads
        self.num_kv_heads = num_kv_heads

        # How many Q heads share each K/V head
        self.groups       = num_heads // num_kv_heads
        self.head_dim     = d_model // num_heads

        # Q projection — full num_heads width
        self.q_proj = nn.Linear(d_model, num_heads * self.head_dim, bias=False)

        # K and V projections — only num_kv_heads width
        self.k_proj = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)

        # Final projection — recombines all heads back into d_model
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        b, seq_len, _ = x.shape

        # Project input to Q, K, V
        Q = self.q_proj(x)  # (b, seq_len, num_heads * head_dim)
        K = self.k_proj(x)  # (b, seq_len, num_kv_heads * head_dim)
        V = self.v_proj(x)  # (b, seq_len, num_kv_heads * head_dim)

        # Reshape into heads
        # Q: (b, num_heads, seq_len, head_dim)
        Q = Q.view(b, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # K, V: (b, num_kv_heads, seq_len, head_dim)
        K = K.view(b, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        V = V.view(b, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE to Q and K — position encoded via rotation, not addition
        # RoPE is NOT applied to V — only Q and K need positional information for attention scores
        freqs = get_rope_frequencies(self.head_dim, seq_len, device=x.device)
        Q     = apply_rope(Q, freqs)
        K     = apply_rope(K, freqs)

        # Expand K and V to match num_heads by repeating each kv head groups times
        # (b, num_kv_heads, seq_len, head_dim) → (b, num_heads, seq_len, head_dim)
        K = K.repeat_interleave(self.groups, dim=1)
        V = V.repeat_interleave(self.groups, dim=1)

        # Scaled dot-product attention scores
        # shape: (b, num_heads, seq_len, seq_len)
        scores = Q @ K.transpose(-2, -1)
        scores = scores / (self.head_dim ** 0.5)

        # Causal mask — prevents each token from attending to future tokens
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device),
            diagonal=1
        ).bool()
        scores = scores.masked_fill(mask, float('-inf'))

        # Convert scores to probabilities — masked positions become 0
        attn_weights = torch.softmax(scores, dim=-1)

        # Weighted sum of values
        context = attn_weights @ V

        # Merge heads back together
        # (b, num_heads, seq_len, head_dim) → (b, seq_len, d_model)
        context = context.transpose(1, 2).contiguous()
        context = context.view(b, seq_len, self.d_model)

        # Final linear projection — mixes information across heads
        return self.out_proj(context)


# ─── Feed Forward Network with SwiGLU ────────────────────────────────────────

class FeedForward(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        # SwiGLU uses 8/3 * d_model for hidden dim instead of 4x
        # round to nearest multiple of d_model for clean tensor shapes
        hidden_dim = int(d_model * 8 / 3)

        # Gate and value projections — run in parallel, multiplied together
        # this gating mechanism gives SwiGLU more expressiveness than plain GELU
        self.w_gate  = nn.Linear(d_model, hidden_dim, bias=False)
        self.w_value = nn.Linear(d_model, hidden_dim, bias=False)

        # Output projection — contracts back to d_model
        self.w_out   = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x):
        # SwiGLU: (Swish(gate) * value) → output
        # Swish is x * sigmoid(x), equivalent to F.silu
        gate   = F.silu(self.w_gate(x))   # Swish activation on gate branch
        value  = self.w_value(x)           # linear value branch
        fused  = gate * value              # element-wise gating
        return self.w_out(fused)


# ─── Transformer Block (GQA + RoPE + RMSNorm + SwiGLU) ───────────────────────

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, num_kv_heads):
        super().__init__()

        self.attention = GroupedQueryAttention(embed_dim, num_heads, num_kv_heads)
        self.ffn       = FeedForward(embed_dim)

        # RMSNorm — Pre-LN position, lighter than LayerNorm
        self.norm1 = RMSNorm(embed_dim)
        self.norm2 = RMSNorm(embed_dim)

    def forward(self, x):
        # Pre-LN: normalise → attention → residual
        x = x + self.attention(self.norm1(x))

        # Pre-LN: normalise → FFN → residual
        x = x + self.ffn(self.norm2(x))

        return x


# ─── Embeddings ───────────────────────────────────────────────────────────────

torch.manual_seed(123)
vocab_size = 10
output_dim = 6

# Token embeddings only — RoPE handles position inside attention
# no positional embedding layer needed
embedding_layer = nn.Embedding(vocab_size, output_dim)
token_ids       = torch.tensor([1, 5, 8])
input_embeddings = embedding_layer(token_ids)

print("Input embeddings (token only — position handled by RoPE):")
print(input_embeddings)
print("Shape:", input_embeddings.shape)


# ─── Test ─────────────────────────────────────────────────────────────────────

torch.manual_seed(123)
block = TransformerBlock(
    embed_dim=6,
    num_heads=6,     # 6 Q heads
    num_kv_heads=2   # 2 KV heads — each shared by 3 Q heads
)

# unsqueeze(0) adds the batch dimension — shape: (1, seq_len, d_model)
output = block(input_embeddings.unsqueeze(0))

print("\nOutput shape:", output.shape)
print("Output:\n", output)