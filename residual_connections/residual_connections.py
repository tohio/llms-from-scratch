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


# ─── Feed Forward Network ─────────────────────────────────────────────────────

class FeedForward(nn.Module):
    def __init__(self, d_model, hidden_dim):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden_dim),  # expand: d_model → hidden_dim
            nn.GELU(),                        # smooth non-linearity
            nn.Linear(hidden_dim, d_model),  # contract: hidden_dim → d_model
        )

    def forward(self, x):
        return self.ffn(x)


# ─── Residual Connection ──────────────────────────────────────────────────────

torch.manual_seed(123)
d_model    = 3
hidden_dim = 12

ffn = FeedForward(d_model=d_model, hidden_dim=hidden_dim)

# Pass input through the sublayer
sublayer_output = ffn(input_embeddings)

# Residual connection — add the original input back to the sublayer output
# this gives the gradient a direct path backwards during training (combats vanishing gradients)
# the model only needs to learn the *difference* (residual) not the full transformation
# output = F(x) + x  — where F(x) is what the sublayer learned to add
residual_output = input_embeddings + sublayer_output

print("Sublayer output:\n", sublayer_output)
print("\nResidual output (sublayer + original input):\n", residual_output)