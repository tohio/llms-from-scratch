import torch
import torch.nn as nn

# ─── Embeddings ───────────────────────────────────────────────────────────────

torch.manual_seed(123)
vocab_size = 10
output_dim = 6

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


# ─── Multi Query Attention ────────────────────────────────────────────────────
# MHA — each head gets its own K and V
# self.k_proj = nn.Linear(d_model, num_heads * head_dim, bias=False)
# self.v_proj = nn.Linear(d_model, num_heads * head_dim, bias=False)

# # MQA — single K and V shared across all heads
# self.k_proj = nn.Linear(d_model, head_dim, bias=False)
# self.v_proj = nn.Linear(d_model, head_dim, bias=False)

class MultiQueryAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model   = d_model
        self.num_heads = num_heads
        self.head_dim  = d_model // num_heads

        # Q projection — full num_heads width, each head gets its own queries
        self.q_proj = nn.Linear(d_model, num_heads * self.head_dim, bias=False)

        # K and V projections — single head only, shared across all Q heads
        # this is the defining characteristic of MQA
        self.k_proj = nn.Linear(d_model, self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self.head_dim, bias=False)

        # Final projection — recombines all heads back into d_model
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        b, seq_len, _ = x.shape

        # Project input to Q, K, V
        Q = self.q_proj(x)  # (b, seq_len, num_heads * head_dim)
        K = self.k_proj(x)  # (b, seq_len, head_dim) — single head
        V = self.v_proj(x)  # (b, seq_len, head_dim) — single head

        # Reshape Q into heads
        # shape: (b, num_heads, seq_len, head_dim)
        Q = Q.view(b, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # K and V stay as single head — just add the head dimension
        # shape: (b, 1, seq_len, head_dim)
        K = K.view(b, seq_len, 1, self.head_dim).transpose(1, 2)
        V = V.view(b, seq_len, 1, self.head_dim).transpose(1, 2)

        # Expand K and V across all Q heads
        # (b, 1, seq_len, head_dim) → (b, num_heads, seq_len, head_dim)
        K = K.expand(b, self.num_heads, seq_len, self.head_dim)
        V = V.expand(b, self.num_heads, seq_len, self.head_dim)

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
mqa = MultiQueryAttention(
    d_model=6,
    num_heads=6
)

output, attention_weights = mqa(input_embeddings.unsqueeze(0))

print("Attention Weights:")
print(attention_weights)
print("\nOutput:")
print(output)


# ─── Parameter Comparison ─────────────────────────────────────────────────────

def count_params(model):
    return sum(p.numel() for p in model.parameters())

mha = MaskedMultiHeadAttention(d_model=6, num_heads=6)
gqa = GroupedQueryAttention(d_model=6, num_heads=6, num_kv_heads=2)

print(f"\nMHA parameters:  {count_params(mha)}")
print(f"GQA parameters:  {count_params(gqa)}")
print(f"MQA parameters:  {count_params(mqa)}")