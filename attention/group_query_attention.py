import torch
import torch.nn as nn

# ─── Embeddings ───────────────────────────────────────────────────────────────

torch.manual_seed(123)
vocab_size = 10
output_dim = 6  # bumped to 6 so we can split into multiple heads cleanly

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


# ─── Grouped Query Attention ──────────────────────────────────────────────────
# MHA — K and V match num_heads exactly
# self.k_proj = nn.Linear(d_model, num_heads * head_dim, bias=False)
# self.v_proj = nn.Linear(d_model, num_heads * head_dim, bias=False)

# # GQA — K and V only need num_kv_heads (smaller than num_heads)
# self.k_proj = nn.Linear(d_model, num_kv_heads * head_dim, bias=False)
# self.v_proj = nn.Linear(d_model, num_kv_heads * head_dim, bias=False)
# The core difference from MHA is two lines:
# K = K.repeat_interleave(self.groups, dim=1)
# V = V.repeat_interleave(self.groups, dim=1)

class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, num_heads, num_kv_heads):
        super().__init__()

        # num_heads must split evenly across kv groups
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        assert num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"

        self.d_model     = d_model
        self.num_heads   = num_heads
        self.num_kv_heads = num_kv_heads

        # How many Q heads share each K/V head
        self.groups      = num_heads // num_kv_heads
        self.head_dim    = d_model // num_heads

        # Q projection — full num_heads width
        self.q_proj = nn.Linear(d_model, num_heads * self.head_dim, bias=False)

        # K and V projections — only num_kv_heads width (smaller than Q)
        # this is the core difference from MHA
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

        # Expand K and V to match num_heads by repeating each kv head `groups` times
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
        # shape: (b, num_heads, seq_len, head_dim)
        context = attn_weights @ V

        # Merge heads back together
        # (b, num_heads, seq_len, head_dim) → (b, seq_len, d_model)
        context = context.transpose(1, 2).contiguous()
        context = context.view(b, seq_len, self.d_model)

        # Final linear projection — mixes information across heads
        output = self.out_proj(context)

        return output, attn_weights


# ─── Test ─────────────────────────────────────────────────────────────────────

torch.manual_seed(123)
gqa = GroupedQueryAttention(
    d_model=6,
    num_heads=6,      # 6 Q heads
    num_kv_heads=2    # 2 KV heads — each shared by 3 Q heads
)

output, attention_weights = gqa(input_embeddings.unsqueeze(0))

print("\nAttention Weights:")
print(attention_weights)
print("\nOutput:")
print(output)

# ─── Comparison ───────────────────────────────────────────────────────────────

# Parameter count comparison — GQA is more memory efficient
# def count_params(model):
#     return sum(p.numel() for p in model.parameters())

# from embeddings.attention import MaskedMultiHeadAttention  

# mha = MaskedMultiHeadAttention(d_model=6, num_heads=6)
# print(f"\nMHA parameters:  {count_params(mha)}")
# print(f"GQA parameters:  {count_params(gqa)}")
# print(f"K/V param reduction: {1 - (gqa.num_kv_heads / gqa.num_heads):.0%}")