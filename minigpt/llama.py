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


# ─── RMS Normalisation ────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-8):
        super().__init__()
        self.eps   = eps

        # Learnable scale parameter — no bias, RMSNorm only scales
        self.scale = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        # Compute root mean square across the embedding dimension
        rms = x.pow(2).mean(dim=-1, keepdim=True).sqrt()

        # Normalise by RMS then apply learnable scale
        return self.scale * x / (rms + self.eps)


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

    # Split x into pairs along the last dimension
    x1 = x[..., 0::2]  # even indices — first of each pair
    x2 = x[..., 1::2]  # odd indices  — second of each pair

    # Compute sin and cos — unsqueeze twice to align with (b, num_heads, seq_len, head_dim/2)
    cos = freqs.cos().unsqueeze(0).unsqueeze(0)
    sin = freqs.sin().unsqueeze(0).unsqueeze(0)

    # Apply rotation to each pair
    # [x1, x2] → [x1·cos(θ) - x2·sin(θ), x1·sin(θ) + x2·cos(θ)]
    rotated_x1 = x1 * cos - x2 * sin
    rotated_x2 = x1 * sin + x2 * cos

    # Interleave rotated pairs back into original shape
    out = torch.stack([rotated_x1, rotated_x2], dim=-1)
    out = out.flatten(-2)

    return out


# ─── Grouped Query Attention with RoPE ───────────────────────────────────────

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
        self.q_proj = nn.Linear(d_model, num_heads * self.head_dim, bias=False)

        # K and V projections — only num_kv_heads width
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

        # Apply RoPE to Q and K — position encoded via rotation not addition
        # RoPE is NOT applied to V — only Q and K need positional info for attention scores
        freqs = get_rope_frequencies(self.head_dim, seq_len, device=x.device)
        Q     = apply_rope(Q, freqs)
        K     = apply_rope(K, freqs)

        # Expand K and V to match num_heads by repeating each kv head groups times
        # (b, num_kv_heads, seq_len, head_dim) → (b, num_heads, seq_len, head_dim)
        K = K.repeat_interleave(self.groups, dim=1)
        V = V.repeat_interleave(self.groups, dim=1)

        # Scaled dot-product attention scores
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
        context = attn_weights @ V

        # Merge heads back together
        # (b, num_heads, seq_len, head_dim) → (b, seq_len, d_model)
        context = context.transpose(1, 2).contiguous()
        context = context.view(b, seq_len, self.d_model)

        # Final linear projection — mixes information across heads
        return self.out_proj(context)


# ─── Feed Forward Network with SwiGLU ────────────────────────────────────────

class FeedForward(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        # SwiGLU uses 8/3 * d_model for hidden dim instead of 4x
        hidden_dim = int(d_model * 8 / 3)

        # Gate and value projections — run in parallel, multiplied together
        # gating mechanism gives SwiGLU more expressiveness than plain GELU
        self.w_gate  = nn.Linear(d_model, hidden_dim, bias=False)
        self.w_value = nn.Linear(d_model, hidden_dim, bias=False)

        # Output projection — contracts back to d_model
        self.w_out   = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x):
        # SwiGLU: (Swish(gate) * value) → output
        # Swish is x * sigmoid(x), equivalent to F.silu
        gate  = F.silu(self.w_gate(x))   # Swish activation on gate branch
        value = self.w_value(x)           # linear value branch
        return self.w_out(gate * value)   # element-wise gating then project


# ─── Transformer Block (LLaMA style) ─────────────────────────────────────────

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, num_kv_heads):
        super().__init__()

        self.attention = GroupedQueryAttention(embed_dim, num_heads, num_kv_heads)
        self.ffn       = FeedForward(embed_dim)

        # RMSNorm — Pre-LN position, lighter than LayerNorm
        # used in LLaMA, Mistral, and most modern architectures
        self.norm1 = RMSNorm(embed_dim)
        self.norm2 = RMSNorm(embed_dim)

    def forward(self, x):
        # Pre-LN: normalise → attention → residual
        x = x + self.attention(self.norm1(x))

        # Pre-LN: normalise → FFN → residual
        x = x + self.ffn(self.norm2(x))

        return x


# ─── MiniLLaMA ────────────────────────────────────────────────────────────────

class MiniLLaMA(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_kv_heads, num_layers):
        super().__init__()

        # Token embedding only — RoPE handles position inside attention
        # no positional embedding layer needed
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)

        # Stack of LLaMA-style transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, num_kv_heads)
            for _ in range(num_layers)
        ])

        # Final RMSNorm before the language model head
        self.final_norm = RMSNorm(embed_dim)

        # Language model head — projects embed_dim → vocab_size to produce logits
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, x):
        # Token embeddings only — no positional embeddings added here
        x = self.token_embedding(x)

        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final normalisation
        x = self.final_norm(x)

        # Project to vocabulary — shape: (b, seq_len, vocab_size)
        logits = self.lm_head(x)

        return logits


# ─── Model ────────────────────────────────────────────────────────────────────

torch.manual_seed(123)

block_size = 4
tokenizer  = tiktoken.encoding_for_model("gpt-4o")

model = MiniLLaMA(
    vocab_size   = tokenizer.n_vocab,
    embed_dim    = 32,
    num_heads    = 4,
    num_kv_heads = 2,   # 2 KV heads — each shared by 2 Q heads
    num_layers   = 2
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

for step, (x, y) in enumerate(train_loader):
    if step >= max_steps:
        break

    x = x.to(device)
    y = y.to(device)

    logits = model(x)

    # Cross entropy loss — measures how well the model predicts the next token
    # logits and targets must be 2D and 1D respectively
    B, T, C = logits.shape
    loss = F.cross_entropy(
        logits.view(B * T, C),
        y.view(B * T)
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 200 == 0:
        print(f"Step {step}, Loss: {loss.item():.4f}")


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