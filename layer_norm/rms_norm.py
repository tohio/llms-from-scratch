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


# ─── RMS Normalisation ────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-8):
        super().__init__()
        self.eps = eps

        # Learnable scale parameter — same as LayerNorm's gamma
        # no bias (beta) — RMSNorm only scales, does not re-center
        self.scale = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        # Compute root mean square across the embedding dimension
        rms = x.pow(2).mean(dim=-1, keepdim=True).sqrt()

        # Normalise by RMS then apply learnable scale
        return self.scale * x / (rms + self.eps)


rms_norm = RMSNorm(d_model)

# Example on the input embeddings
normalized = rms_norm(input_embeddings)

print("Original input embeddings:")
print(input_embeddings)

# After RMSNorm scale is controlled but mean is NOT forced to 0
# RMSNorm skips the mean centering step that LayerNorm performs
print("\nAfter RMSNorm (scale controlled, mean not centred):")
print(normalized)

# RMS per token should be ≈ 1, mean is not guaranteed to be ≈ 0
print("Mean per token:", normalized.mean(dim=-1))
print("RMS per token: ", normalized.pow(2).mean(dim=-1).sqrt())