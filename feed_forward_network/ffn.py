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

        # Two linear layers with GELU activation in between
        # hidden_dim is typically 4x d_model — expands then contracts the representation
        # this gives the model capacity to learn non-linear transformations per token
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden_dim),  # expand: d_model → hidden_dim
            nn.GELU(),                        # smooth non-linearity — modern GPTs use GELU over ReLU
            nn.Linear(hidden_dim, d_model),  # contract: hidden_dim → d_model
        )

    def forward(self, x):
        # Applied independently to each token position
        # shape in == shape out: (seq_len, d_model)
        return self.ffn(x)


# ─── Test ─────────────────────────────────────────────────────────────────────

torch.manual_seed(123)
d_model    = 3
hidden_dim = 12  # 4x d_model — standard expansion ratio

ffn = FeedForward(d_model=d_model, hidden_dim=hidden_dim)
output_after_ffn = ffn(input_embeddings)

print("\nInput to FFN (shape):", input_embeddings.shape)
print("Output after FFN (shape):", output_after_ffn.shape)
print("Output:\n", output_after_ffn)