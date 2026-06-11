import torch
import torch.nn as nn

# ─── Embeddings ───────────────────────────────────────────────────────────────

torch.manual_seed(123)
vocab_size = 10
output_dim = 3

# Token embeddings — each token ID gets a learnable vector
embedding_layer = nn.Embedding(vocab_size, output_dim)
token_ids = torch.tensor([1, 5, 8])
token_embeddings = embedding_layer(token_ids)

# Positional embeddings — encodes the position of each token in the sequence
max_length = 3
pos_embedding_layer = nn.Embedding(max_length, output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(max_length))

# Final input — token meaning + position information combined
input_embeddings = token_embeddings + pos_embeddings
print("Input embeddings (token + position):")
print(input_embeddings)
print("Shape:", input_embeddings.shape)


# ─── Masked Multi-Head Attention ──────────────────────────────────────────────

class MaskedMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()

        # d_model must split evenly across heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model   = d_model
        self.num_heads = num_heads

        # Each head works on a slice of the full embedding dimension
        self.head_dim  = d_model // num_heads

        # QKV projections — same as single head but full d_model width
        # the split into heads happens after projection
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)

        # Final projection — recombines all heads back into d_model
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        b, seq_len, _ = x.shape

        # Project input to Q, K, V — shape: (b, seq_len, d_model)
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Split d_model into num_heads × head_dim
        # shape: (b, seq_len, num_heads, head_dim)
        Q = Q.view(b, seq_len, self.num_heads, self.head_dim)
        K = K.view(b, seq_len, self.num_heads, self.head_dim)
        V = V.view(b, seq_len, self.num_heads, self.head_dim)

        # Move heads before seq_len so each head can compute attention independently
        # shape: (b, num_heads, seq_len, head_dim)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Scaled dot-product attention scores
        # shape: (b, num_heads, seq_len, seq_len)
        scores = Q @ K.transpose(-2, -1)
        scores = scores / (self.head_dim ** 0.5)

        # Causal mask — prevents each token from attending to future tokens
        # upper triangle is True, those positions get -inf before softmax
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device),
            diagonal=1
        ).bool()
        scores = scores.masked_fill(mask, float('-inf'))

        # Convert scores to probabilities — masked positions become 0
        attn_weights = torch.softmax(scores, dim=-1)

        # Weighted sum of values
        # shape: (b, num_heads, seq_len, head_dim)
        context = attn_weights @ V

        # Merge heads back together
        # (b, num_heads, seq_len, head_dim) → (b, seq_len, num_heads, head_dim)
        context = context.transpose(1, 2).contiguous()

        # Flatten heads into d_model
        # shape: (b, seq_len, d_model)
        context = context.view(b, seq_len, self.d_model)

        # Final linear projection — mixes information across heads
        output = self.out_proj(context)

        return output, attn_weights


# ─── Test ─────────────────────────────────────────────────────────────────────

torch.manual_seed(123)
mha = MaskedMultiHeadAttention(
    d_model=3,
    num_heads=3
)

# unsqueeze(0) adds the batch dimension — shape goes from (seq_len, d_model) to (1, seq_len, d_model)
output, attention_weights = mha(input_embeddings.unsqueeze(0))

print("\nAttention Weights:")
print(attention_weights)
print("\nOutput:")
print(output)