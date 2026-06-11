import torch
import torch.nn as nn

# ─── Embeddings ───────────────────────────────────────────────────────────────

torch.manual_seed(123)
vocab_size = 10
output_dim = 3
d_model    = 3

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


# ─── Layer Normalisation ──────────────────────────────────────────────────────

# LayerNorm normalises across the embedding dimension (d_model) for each token
# keeps activations stable as they flow through many layers
# without it, values can explode or vanish as the network gets deeper
ln = nn.LayerNorm(d_model)

normalized = ln(input_embeddings)

print("\nOriginal input embeddings:")
print(input_embeddings)

# After LayerNorm each token vector has mean ≈ 0 and std ≈ 1
# this is computed independently per token — not across the batch or sequence
print("\nAfter LayerNorm (mean ≈ 0, std ≈ 1 per token):")
print(normalized)

# Verify — mean and std are computed across the last dimension (d_model)
print("\nMean per token (should be ≈ 0):", normalized.mean(dim=-1))
print("Std per token  (should be ≈ 1):", normalized.std(dim=-1, unbiased=False))