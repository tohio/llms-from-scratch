import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tiktoken
from torch.utils.data import Dataset, DataLoader


# ─── Hardware Config ──────────────────────────────────────────────────────────

device = torch.device(
    "mps"  if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available()         else
    "cpu"
)

print(f"Using device: {device}")

# ── Default — tiny_corpus.txt ──
# Sized to match the tiny corpus (~700 words)
# A larger model would immediately overfit this data
# Swap to the curated presets below when using fineweb/dolma corpora
embed_dim    = 32
num_heads    = 4
num_kv_heads = 2    # 2 KV heads — each shared by 2 Q heads
num_layers   = 2
block_size   = 4
max_steps  = 5000

# ── Curated corpus (fineweb_corpus.txt / dolma_corpus.txt) ──
# Use these when swapping to a larger curated corpus from data_curation/
# ── Laptop / M4 Max ──
# embed_dim    = 256
# num_heads    = 8
# num_kv_heads = 4
# num_layers   = 8
# block_size   = 128
# max_steps  = 50000

# ── Cloud GPU (A100/H100) ──
# embed_dim    = 512
# num_heads    = 16
# num_kv_heads = 8
# num_layers   = 12
# block_size   = 256
# USE_COMPILE  = True    # torch.compile — significant speedup on CUDA
# USE_AMP      = True    # automatic mixed precision — CUDA only, not MPS
# max_steps  = 200000

# ── CPU / Small GPU (Tesla V100)──
# embed_dim    = 128
# num_heads    = 4
# num_kv_heads = 2
# num_layers   = 4
# block_size   = 64
# max_steps  = 20000


# ─── Config ───────────────────────────────────────────────────────────────────

# ── Default — tiny_corpus.txt ──
CORPUS_PATH = "../data/tiny_corpus.txt"

# ── FineWeb curated corpus ──
# CORPUS_PATH = "../data/fineweb_corpus.txt"

# ── Dolma curated corpus ──
# CORPUS_PATH = "../data/dolma_corpus.txt"

# ── Mixed curated corpus ──
# CORPUS_PATH = "../data/mixed_corpus.txt"

learning_rate = 3e-4
batch_size    = 32



# ─── Tokenizer ────────────────────────────────────────────────────────────────

tokenizer = tiktoken.encoding_for_model("gpt-4o")


# ─── Dataset ──────────────────────────────────────────────────────────────────

class TinyCorpusDataset(Dataset):
    def __init__(self, path, block_size):
        # Load the raw text from disk
        with open(path, "r", encoding="utf8") as f:
            text = f.read()

        # Tokenize the entire corpus into a flat list of integer token IDs
        # using GPT-4o's BPE tokenizer
        data = tokenizer.encode(text)

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


# ─── RoPE ─────────────────────────────────────────────────────────────────────

def get_rope_frequencies(head_dim, seq_len, base=10000, device='cpu'):
    # Compute theta for each dimension pair: θ = 1 / (10000 ^ (2i / head_dim))
    thetas    = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim)).to(device)
    # Positions in the sequence [0, 1, 2, ..., seq_len-1]
    positions = torch.arange(seq_len).float().to(device)
    # Outer product — every position multiplied by every theta
    # shape: (seq_len, head_dim/2)
    return torch.outer(positions, thetas)


def apply_rope(x, freqs):
    # x shape: (b, num_heads, seq_len, head_dim)
    # freqs shape: (seq_len, head_dim/2)
    x1 = x[..., 0::2]  # even indices — first of each pair
    x2 = x[..., 1::2]  # odd indices  — second of each pair

    cos = freqs.cos().unsqueeze(0).unsqueeze(0)
    sin = freqs.sin().unsqueeze(0).unsqueeze(0)

    # Apply rotation: [x1, x2] → [x1·cos - x2·sin, x1·sin + x2·cos]
    rotated_x1 = x1 * cos - x2 * sin
    rotated_x2 = x1 * sin + x2 * cos

    out = torch.stack([rotated_x1, rotated_x2], dim=-1)
    return out.flatten(-2)


# ─── Grouped Query Attention with RoPE and F.scaled_dot_product_attention ─────

class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, num_heads, num_kv_heads):
        super().__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        assert num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"

        self.d_model      = d_model
        self.num_heads    = num_heads
        self.num_kv_heads = num_kv_heads
        # How many Q heads share each K/V head
        self.groups       = num_heads // num_kv_heads
        self.head_dim     = d_model // num_heads

        # Q projection — full num_heads width
        self.q_proj   = nn.Linear(d_model, num_heads * self.head_dim,    bias=False)
        # K and V projections — only num_kv_heads width
        self.k_proj   = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.v_proj   = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        b, seq_len, _ = x.shape

        Q = self.q_proj(x).view(b, seq_len, self.num_heads,    self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(b, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(b, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE to Q and K — not V
        freqs = get_rope_frequencies(self.head_dim, seq_len, device=x.device)
        Q     = apply_rope(Q, freqs)
        K     = apply_rope(K, freqs)

        # Expand K and V to match num_heads — each KV head shared across groups
        K = K.repeat_interleave(self.groups, dim=1)
        V = V.repeat_interleave(self.groups, dim=1)

        # F.scaled_dot_product_attention — fused attention kernel
        # is_causal=True handles the causal mask internally — no manual mask needed
        # uses FlashAttention under the hood when available for significant speedup
        context = F.scaled_dot_product_attention(Q, K, V, is_causal=True)

        context = context.transpose(1, 2).contiguous().view(b, seq_len, self.d_model)
        return self.out_proj(context)


# ─── Feed Forward Network with SwiGLU ────────────────────────────────────────

class FeedForward(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        # SwiGLU uses 8/3 * d_model for hidden dim instead of 4x
        hidden_dim   = int(d_model * 8 / 3)
        self.w_gate  = nn.Linear(d_model, hidden_dim, bias=False)
        self.w_value = nn.Linear(d_model, hidden_dim, bias=False)
        self.w_out   = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x):
        # SwiGLU: Swish(gate) * value → output
        # F.silu is the PyTorch native Swish — x * sigmoid(x)
        gate  = F.silu(self.w_gate(x))
        value = self.w_value(x)
        return self.w_out(gate * value)


# ─── Transformer Block (LLaMA style — PyTorch native) ────────────────────────

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, num_kv_heads):
        super().__init__()
        self.attention = GroupedQueryAttention(embed_dim, num_heads, num_kv_heads)
        self.ffn       = FeedForward(embed_dim)
        # nn.RMSNorm — PyTorch native RMSNorm added in PyTorch 2.4
        # replaces the custom RMSNorm class used in minigpt/llama.py
        self.norm1 = nn.RMSNorm(embed_dim)
        self.norm2 = nn.RMSNorm(embed_dim)

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


# ─── MiniLLaMA (PyTorch Native) ──────────────────────────────────────────────

class MiniLLaMA(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_kv_heads, num_layers):
        super().__init__()
        # Token embedding only — RoPE handles position inside attention
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, num_kv_heads)
            for _ in range(num_layers)
        ])
        # Final RMSNorm — PyTorch native
        self.final_norm = nn.RMSNorm(embed_dim)
        self.lm_head    = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, x):
        x = self.token_embedding(x)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        return self.lm_head(x)


# ─── Training ─────────────────────────────────────────────────────────────────

torch.manual_seed(123)

model = MiniLLaMA(
    vocab_size   = tokenizer.n_vocab,
    embed_dim    = embed_dim,
    num_heads    = num_heads,
    num_kv_heads = num_kv_heads,
    num_layers   = num_layers
).to(device)

print(model)

dataset       = TinyCorpusDataset(CORPUS_PATH, block_size=block_size)
split_idx     = int(0.9 * len(dataset))
train_dataset = torch.utils.data.Subset(dataset, range(0, split_idx))
val_dataset   = torch.utils.data.Subset(dataset, range(split_idx, len(dataset)))
train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader    = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)

print(f"Train sequences:      {len(train_dataset)}")
print(f"Validation sequences: {len(val_dataset)}")
print(f"Batches per epoch:    {len(train_loader)}")

optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

model.train()
step = 0

# Outer loop keeps cycling through the dataloader until max_steps is reached
# without this the model only sees the data once (27 batches) not 5000 steps
while step < max_steps:
    for x, y in train_loader:
        if step >= max_steps:
            break

        x, y    = x.to(device), y.to(device)
        logits  = model(x)
        B, T, C = logits.shape
        loss    = F.cross_entropy(logits.view(B * T, C), y.view(B * T))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 200 == 0:
            print(f"Step {step}, Loss: {loss.item():.4f}")

        step += 1


# ─── Validation ───────────────────────────────────────────────────────────────

model.eval()
val_loss  = 0.0
val_steps = 0

with torch.no_grad():
    for x, y in val_loader:
        x, y    = x.to(device), y.to(device)
        logits  = model(x)
        B, T, C = logits.shape
        loss    = F.cross_entropy(logits.view(B * T, C), y.view(B * T))
        val_loss  += loss.item()
        val_steps += 1

print(f"Validation Loss: {val_loss / val_steps:.4f}")


# ─── Generation ───────────────────────────────────────────────────────────────

def generate(model, prompt, max_new_tokens=50):
    model.eval()
    tokens = tokenizer.encode(prompt)
    x      = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            x_cond     = x[:, -block_size:]
            logits     = model(x_cond)
            logits     = logits[:, -1, :]
            probs      = torch.softmax(logits, dim=-1)
            next_token = torch.argmax(probs, dim=-1, keepdim=True)
            x          = torch.cat([x, next_token], dim=1)

    return tokenizer.decode(x[0].tolist())


# ─── Test Generation ──────────────────────────────────────────────────────────

prompt = "The lighthouse"

print(generate(model, prompt, max_new_tokens=15))
print(generate(model, prompt, max_new_tokens=50))