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
seq_length = 3
pos_embedding_layer = nn.Embedding(seq_length, output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(seq_length))

# Final input — token meaning + position information combined
input_embeddings = token_embeddings + pos_embeddings
print("Input embeddings (token + position):")
print(input_embeddings)
print("Shape:", input_embeddings.shape)

# ─── Raw Attention (step by step) ─────────────────────────────────────────────

input_dim = 3
torch.manual_seed(123)

# Project input embeddings into Query, Key, Value spaces
W_query = nn.Linear(input_dim, output_dim, bias=False)
W_key   = nn.Linear(input_dim, output_dim, bias=False)
W_value = nn.Linear(input_dim, output_dim, bias=False)

queries = W_query(input_embeddings)
keys    = W_key(input_embeddings)
values  = W_value(input_embeddings)

print("\nQueries:\n", queries)
print("Keys:\n", keys)
print("Values:\n", values)

# Raw dot product between queries and keys — how similar is each token to every other
attention_scores = queries @ keys.T
print("\nAttention scores:\n", attention_scores)

# Scale by sqrt(dim) to prevent large dot products from pushing softmax into flat regions
scale = keys.shape[-1] ** 0.5
scaled_scores = attention_scores / scale
print("\nScaled scores:\n", scaled_scores)

# Convert scores to probabilities — each row sums to 1
attention_weights = torch.softmax(scaled_scores, dim=-1)
print("\nAttention weights:\n", attention_weights)
print("Row sums (should be 1):", attention_weights.sum(dim=-1))

# Weighted sum of values — the new token representation after attending to context
context_vectors = attention_weights @ values
print("\nContext vectors:\n", context_vectors)


# ─── Self Attention Class ─────────────────────────────────────────────────────

class SelfAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

        # Three linear layers that project input → Q, K, V
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Scaled dot-product attention
        scores       = Q @ K.T / (self.d_model ** 0.5)
        attn_weights = torch.softmax(scores, dim=-1)
        context_vec  = attn_weights @ V

        return context_vec, attn_weights


torch.manual_seed(123)
sa = SelfAttention(d_model=output_dim)
context_vectors, attention_weights = sa(input_embeddings)

print("\nAttention weights (how much each token attends to every other token):")
print(attention_weights)
print("\nContext vectors (the new representation after attention):")
print(context_vectors)


# ─── Causal Self Attention Class ──────────────────────────────────────────────

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

        # Three linear layers that project input → Q, K, V
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Scaled dot-product attention
        scores = Q @ K.T / (self.d_model ** 0.5)

        # Causal mask — upper triangle is 1, prevents attending to future tokens
        # token at position i can only attend to positions 0..i
        seq_len = x.shape[0]
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)

        # Replace future positions with -inf so softmax drives them to 0
        scores = scores.masked_fill(mask == 1, float('-inf'))

        # Convert scores into probabilities
        attn_weights = torch.softmax(scores, dim=-1)

        # Weighted sum of values
        context_vec = attn_weights @ V

        return context_vec, attn_weights


torch.manual_seed(123)
causal_sa = CausalSelfAttention(d_model=output_dim)
context_vectors, attention_weights = causal_sa(input_embeddings)

print("\nCausal Attention Weights:")
print(attention_weights)
print("\nCausal Context Vectors:")
print(context_vectors)