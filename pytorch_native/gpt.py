import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import tiktoken
from torch.utils.data import Dataset, DataLoader

# ─── Dataset ──────────────────────────────────────────────────────────────────

class TinyCorpusDataset(Dataset):
    def __init__(self, path, block_size):
        # Load the raw text from disk
        with open(path, "r", encoding="utf8") as f:
            text = f.read()

        # Tokenize the entire corpus into a flat list of integer token IDs
        # using GPT-4o's BPE tokenizer
        tokenizer = tiktoken.encoding_for_model("gpt-4o")
        data      = tokenizer.encode(text)

        # Store as a tensor for efficient indexing during training
        self.data       = torch.tensor(data, dtype=torch.long)
        self.block_size = block_size

    def __len__(self):
        # Total number of valid sequences we can extract from the corpus
        # each sequence needs block_size tokens for x and block_size tokens for y
        # so the last valid start index is len(data) - block_size
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        # x is the input sequence — block_size tokens starting at idx
        # y is the target sequence — same length but shifted one position right
        # the model learns to predict y[t] given x[0..t]
        x = self.data[idx:idx + self.block_size]
        y = self.data[idx + 1:idx + self.block_size + 1]
        return x, y


# ─── MiniGPT ──────────────────────────────────────────────────────────────────

class MiniGPT(nn.Module):
    def __init__(self, vocab_size, block_size, embed_dim, num_heads, hidden_dim, num_layers):
        super().__init__()

        # Token embedding — maps token IDs to vectors
        self.token_embedding    = nn.Embedding(vocab_size, embed_dim)

        # Positional embedding — encodes position of each token in the sequence
        self.position_embedding = nn.Embedding(block_size, embed_dim)

        # TransformerEncoderLayer — self attention + FFN, no cross attention
        # this is the correct building block for decoder-only GPT style models
        # norm_first=True — Pre-LN for training stability
        # is_causal=True passed at forward time — prevents attending to future tokens
        self.block = nn.TransformerEncoderLayer(
            d_model        = embed_dim,
            nhead          = num_heads,
            dim_feedforward = hidden_dim,
            activation     = "gelu",
            norm_first     = True,
            batch_first    = True,
            dropout        = 0.0,
            enable_nested_tensor=False
        )

        # Stack num_layers blocks
        self.blocks = nn.TransformerEncoder(self.block, num_layers=num_layers)

        # Final normalisation before the language model head
        self.final_norm = nn.LayerNorm(embed_dim)

        # Language model head — projects embed_dim → vocab_size to produce logits
        # logits are unnormalised scores over the vocabulary for the next token prediction
        self.lm_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        b, seq_len = x.shape

        # Combine token and positional embeddings
        token_embeddings    = self.token_embedding(x)
        positions           = torch.arange(seq_len, device=x.device)
        position_embeddings = self.position_embedding(positions)
        x = token_embeddings + position_embeddings

        # Generate causal mask — upper triangle is -inf, prevents attending to future tokens
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=x.device)

        # Pass through transformer blocks with causal mask
        x = self.blocks(x, mask=mask, is_causal=True)

        # Final normalisation
        x = self.final_norm(x)

        # Project to vocabulary — shape: (b, seq_len, vocab_size)
        logits = self.lm_head(x)

        return logits


# ─── Model ────────────────────────────────────────────────────────────────────

torch.manual_seed(123)

block_size = 4
tokenizer  = tiktoken.encoding_for_model("gpt-4o")

model = MiniGPT(
    vocab_size  = tokenizer.n_vocab,
    block_size  = block_size,
    embed_dim   = 32,
    num_heads   = 4,
    hidden_dim  = 128,
    num_layers  = 2
)

print(model)

device = "cuda" if torch.cuda.is_available() else "cpu"
model  = model.to(device)


# ─── Training ─────────────────────────────────────────────────────────────────

learning_rate = 3e-4
batch_size    = 32
max_steps     = 5000

# Build dataset and dataloader
dataset = TinyCorpusDataset("../data/tiny_corpus.txt", block_size=block_size)

# 90/10 train/validation split
split_idx     = int(0.9 * len(dataset))
train_dataset = torch.utils.data.Subset(dataset, range(0, split_idx))
val_dataset   = torch.utils.data.Subset(dataset, range(split_idx, len(dataset)))

train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader    = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)

print(f"Train sequences:      {len(train_dataset)}")
print(f"Validation sequences: {len(val_dataset)}")
print(f"Batches per epoch:    {len(train_loader)}")

# AdamW — Adam with weight decay, standard for transformer training
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

model.train()
step = 0

# outer loop keeps cycling through the dataloader until max_steps is reached
# without this the model only sees the data once (27 batches) not 5000 steps
while step < max_steps:
    for x, y in train_loader:
        if step >= max_steps:
            break

        x = x.to(device)
        y = y.to(device)

        logits  = model(x)
        B, T, C = logits.shape

        # Cross entropy loss — measures how well the model predicts the next token
        # logits and targets must be 2D and 1D respectively
        loss = F.cross_entropy(
            logits.view(B * T, C),
            y.view(B * T)
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 200 == 0:
            print(f"Step {step}, Loss: {loss.item():.4f}")

        step += 1


# ─── Generation ───────────────────────────────────────────────────────────────

def generate(model, tokenizer, prompt, max_new_tokens=50):
    model.eval()
    device = next(model.parameters()).device

    # Encode prompt and add batch dimension
    tokens = tokenizer.encode(prompt)
    x      = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Crop to block_size — model can only see block_size tokens at a time
            x_cond = x[:, -block_size:]
            logits = model(x_cond)

            # Take logits for the last token position only
            logits     = logits[:, -1, :]
            probs      = torch.softmax(logits, dim=-1)

            # Greedy decoding — always pick the most likely next token
            next_token = torch.argmax(probs, dim=-1, keepdim=True)
            x          = torch.cat([x, next_token], dim=1)

    return tokenizer.decode(x[0].tolist())


# ─── Test Generation ──────────────────────────────────────────────────────────

prompt = "The lighthouse"

print(generate(model, tokenizer, prompt, max_new_tokens=15))
print(generate(model, tokenizer, prompt, max_new_tokens=50))