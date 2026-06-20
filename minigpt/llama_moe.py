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
num_experts  = 8    # 8 experts per layer — same as LLaMA 4 Scout
top_k        = 2    # each token routed to 2 experts
block_size   = 4

# ── Curated corpus (fineweb_corpus.txt / dolma_corpus.txt) ──
# Use these when swapping to a larger curated corpus from data_curation/
# ── Laptop / M4 Max ──
# embed_dim    = 256
# num_heads    = 8
# num_kv_heads = 4
# num_layers   = 8
# num_experts  = 8
# top_k        = 2
# block_size   = 128

# ── Cloud GPU (A100/H100) ──
# embed_dim    = 512
# num_heads    = 16
# num_kv_heads = 8
# num_layers   = 12
# num_experts  = 8
# top_k        = 2
# block_size   = 256
# USE_COMPILE  = True    # torch.compile — significant speedup on CUDA
# USE_AMP      = True    # automatic mixed precision — CUDA only, not MPS

# ── CPU / Small GPU ──
# embed_dim    = 128
# num_heads    = 4
# num_kv_heads = 2
# num_layers   = 4
# num_experts  = 8
# top_k        = 2
# block_size   = 64


# ─── Config ───────────────────────────────────────────────────────────────────

# Swap this path when using curated corpora from data_curation/
CORPUS_PATH   = "../data/tiny_corpus.txt"

learning_rate = 3e-4
batch_size    = 32
max_steps     = 5000


# ─── Tokenizer ────────────────────────────────────────────────────────────────

tokenizer = tiktoken.encoding_for_model("gpt-4o")


# ─── Dataset ──────────────────────────────────────────────────────────────────

class TinyCorpusDataset(Dataset):
    def __init__(self, path, block_size):
        with open(path, "r", encoding="utf8") as f:
            text = f.read()

        data = tokenizer.encode(text)

        self.data       = torch.tensor(data, dtype=torch.long)
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.block_size]
        y = self.data[idx + 1:idx + self.block_size + 1]
        return x, y


# ─── RoPE ─────────────────────────────────────────────────────────────────────

def get_rope_frequencies(head_dim, seq_len, base=10000, device='cpu'):
    thetas    = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim)).to(device)
    positions = torch.arange(seq_len).float().to(device)
    return torch.outer(positions, thetas)


def apply_rope(x, freqs):
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    cos = freqs.cos().unsqueeze(0).unsqueeze(0)
    sin = freqs.sin().unsqueeze(0).unsqueeze(0)
    rotated_x1 = x1 * cos - x2 * sin
    rotated_x2 = x1 * sin + x2 * cos
    out = torch.stack([rotated_x1, rotated_x2], dim=-1)
    return out.flatten(-2)


# ─── Grouped Query Attention with RoPE ───────────────────────────────────────

class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, num_heads, num_kv_heads):
        super().__init__()

        assert d_model % num_heads == 0
        assert num_heads % num_kv_heads == 0

        self.d_model      = d_model
        self.num_heads    = num_heads
        self.num_kv_heads = num_kv_heads
        self.groups       = num_heads // num_kv_heads
        self.head_dim     = d_model // num_heads

        self.q_proj   = nn.Linear(d_model, num_heads * self.head_dim,    bias=False)
        self.k_proj   = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.v_proj   = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        b, seq_len, _ = x.shape

        Q = self.q_proj(x).view(b, seq_len, self.num_heads,    self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(b, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(b, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        freqs = get_rope_frequencies(self.head_dim, seq_len, device=x.device)
        Q     = apply_rope(Q, freqs)
        K     = apply_rope(K, freqs)

        K = K.repeat_interleave(self.groups, dim=1)
        V = V.repeat_interleave(self.groups, dim=1)

        scores = Q @ K.transpose(-2, -1) / (self.head_dim ** 0.5)
        mask   = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(mask, float('-inf'))
        attn_weights = torch.softmax(scores, dim=-1)
        context = attn_weights @ V

        context = context.transpose(1, 2).contiguous().view(b, seq_len, self.d_model)
        return self.out_proj(context)


# ─── Expert ───────────────────────────────────────────────────────────────────

class Expert(nn.Module):
    def __init__(self, d_model, hidden_dim):
        super().__init__()
        # Each expert is an independent SwiGLU FFN
        # same architecture as LLaMA's FeedForward — only the routing changes
        self.w_gate  = nn.Linear(d_model, hidden_dim, bias=False)
        self.w_value = nn.Linear(d_model, hidden_dim, bias=False)
        self.w_out   = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x):
        gate  = F.silu(self.w_gate(x))
        value = self.w_value(x)
        return self.w_out(gate * value)


# ─── Router ───────────────────────────────────────────────────────────────────

class Router(nn.Module):
    def __init__(self, d_model, num_experts):
        super().__init__()
        self.gate = nn.Linear(d_model, num_experts, bias=False)

    def forward(self, x, top_k):
        logits  = self.gate(x)
        probs   = torch.softmax(logits, dim=-1)
        weights, indices = torch.topk(probs, k=top_k, dim=-1)
        # Renormalise weights so they sum to 1 across selected experts
        weights = weights / weights.sum(dim=-1, keepdim=True)
        return weights, indices


# ─── Mixture of Experts FFN ───────────────────────────────────────────────────

class MoE(nn.Module):
    def __init__(self, d_model, hidden_dim, num_experts, top_k):
        super().__init__()

        assert top_k <= num_experts, "top_k cannot exceed num_experts"

        self.d_model     = d_model
        self.num_experts = num_experts
        self.top_k       = top_k

        # Pool of expert FFNs — each learns different transformations
        self.experts = nn.ModuleList([
            Expert(d_model, hidden_dim)
            for _ in range(num_experts)
        ])

        # Router — decides which experts handle each token
        self.router = Router(d_model, num_experts)

    def forward(self, x):
        b, seq_len, d_model = x.shape

        # Flatten batch and sequence dimensions for routing
        x_flat = x.view(-1, d_model)

        weights, indices = self.router(x_flat, self.top_k)

        output = torch.zeros_like(x_flat)

        for k in range(self.top_k):
            expert_idx = indices[:, k]
            weight     = weights[:, k]

            for i in range(self.num_experts):
                token_mask = (expert_idx == i)
                if token_mask.any():
                    expert_output       = self.experts[i](x_flat[token_mask])
                    output[token_mask] += weight[token_mask].unsqueeze(-1) * expert_output

        return output.view(b, seq_len, d_model)


# ─── Transformer Block (LLaMA 4 style — GQA + RoPE + RMSNorm + MoE) ─────────

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, num_kv_heads, num_experts, top_k):
        super().__init__()

        self.attention = GroupedQueryAttention(embed_dim, num_heads, num_kv_heads)

        # MoE replaces the standard FFN — same slot, different internals
        # hidden_dim follows LLaMA convention of 8/3 * d_model per expert
        self.moe = MoE(
            d_model     = embed_dim,
            hidden_dim  = int(embed_dim * 8 / 3),
            num_experts = num_experts,
            top_k       = top_k
        )

        # RMSNorm — Pre-LN position, lighter than LayerNorm
        self.norm1 = nn.RMSNorm(embed_dim)
        self.norm2 = nn.RMSNorm(embed_dim)

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        # MoE sits in the exact same position as a standard FFN
        x = x + self.moe(self.norm2(x))
        return x


# ─── MiniLLaMAMoE ────────────────────────────────────────────────────────────

class MiniLLaMAMoE(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_kv_heads, num_layers, num_experts, top_k):
        super().__init__()

        # Token embedding only — RoPE handles position inside attention
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, num_kv_heads, num_experts, top_k)
            for _ in range(num_layers)
        ])

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

model = MiniLLaMAMoE(
    vocab_size   = tokenizer.n_vocab,
    embed_dim    = embed_dim,
    num_heads    = num_heads,
    num_kv_heads = num_kv_heads,
    num_layers   = num_layers,
    num_experts  = num_experts,
    top_k        = top_k
).to(device)

print(model)

# Print parameter breakdown
total_params  = sum(p.numel() for p in model.parameters())
active_params = sum(p.numel() for p in model.blocks[0].moe.experts[0].parameters()) * top_k
print(f"\nTotal parameters:         {total_params:,}")
print(f"Active MoE params/token:  {active_params:,} ({100 * active_params / total_params:.1f}% of total)")

dataset       = TinyCorpusDataset(CORPUS_PATH, block_size=block_size)
split_idx     = int(0.9 * len(dataset))
train_dataset = torch.utils.data.Subset(dataset, range(0, split_idx))
val_dataset   = torch.utils.data.Subset(dataset, range(split_idx, len(dataset)))
train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader    = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)

print(f"\nTrain sequences:      {len(train_dataset)}")
print(f"Validation sequences: {len(val_dataset)}")
print(f"Batches per epoch:    {len(train_loader)}")

optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

model.train()
step = 0

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