import torch
import torch.nn as nn

# Hyperparameters
vocab_size = 10
output_dim = 3
seq_length = 3

torch.manual_seed(123)

# Token embeddings — each token ID gets a learnable vector of size output_dim
embedding_layer = nn.Embedding(vocab_size, output_dim)

# Look up embeddings for a sequence of token IDs
token_embeddings = embedding_layer(torch.tensor([1, 5, 8]))
print("Token embeddings shape:", token_embeddings.shape)
print("Token embeddings:\n", token_embeddings)

# Positional embeddings — encodes the position of each token in the sequence
# torch.arange(seq_length) produces [0, 1, 2] — one position per token
pos_embedding_layer = nn.Embedding(seq_length, output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(seq_length))
print("\nPositional embeddings shape:", pos_embeddings.shape)
print("Positional embeddings:\n", pos_embeddings)

# Final input embeddings — token meaning + position information combined
# the model now knows both what each token is and where it sits in the sequence
input_embeddings = token_embeddings + pos_embeddings
print("\nInput embeddings shape:", input_embeddings.shape)
print("Input embeddings:\n", input_embeddings)