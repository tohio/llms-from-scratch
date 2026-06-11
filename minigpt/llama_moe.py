import torch
import torch.nn as nn
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
        # SwiGLU: (Swish(gate) * value) → output
        gate  = F.silu(self.w_gate(x))
        value = self.w_value(x)
        return self.w_out(gate * value)


# ─── Router ───────────────────────────────────────────────────────────────────

class Router(nn.Module):
    def __init__(self, d_model, num_experts):
        super().__init__()

        # Linear layer that scores each token against each expert
        self.gate = nn.Linear(d_model, num_experts, bias=False)

    def forward(self, x, top_k):
        # Compute raw scores for each expert
        logits = self.gate(x)

        # Convert to probabilities
        probs = torch.softmax(logits, dim=-1)

        # Select top_k experts per token
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
        # shape: (batch * seq_len, d_model)
        x_flat = x.view(-1, d_model)

        # Get expert weights and indices for each token
        # weights shape: (batch * seq_len, top_k)
        # indices shape: (batch * seq_len, top_k)
        weights, indices = self.router(x_flat, self.top_k)

        # Accumulate weighted expert outputs
        output = torch.zeros_like(x_flat)

        for k in range(self.top_k):
            expert_idx = indices[:, k]   # (batch * seq_len,)
            weight     = weights[:, k]   # (batch * seq_len,)

            for i in range(self.num_experts):
                # Find which tokens are assigned to expert i for this slot
                token_mask = (expert_idx == i)

                if token_mask.any():
                    expert_output          = self.experts[i](x_flat[token_mask])
                    output[token_mask]    += weight[token_mask].unsqueeze(-1) * expert_output

        # Restore original shape
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
        # Pre-LN: normalise → attention → residual
        x = x + self.attention(self.norm1(x))

        # Pre-LN: normalise → MoE FFN → residual
        # MoE sits in the exact same position as a standard FFN
        x = x + self.moe(self.norm2(x))

        return x


# ─── MiniLLaMA MoE ────────────────────────────────────────────────────────────

class MiniLLaMAMoE(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_kv_heads, num_layers, num_experts, top_k):
        super().__init__()

        # Token embedding only — RoPE handles position inside attention
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)

        # Stack of LLaMA 4 style transformer blocks with MoE FFN
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, num_kv_heads, num_experts, top_k)
            for _ in range(num_layers)
        ])

        # Final RMSNorm before the language model head
        self.final_norm = nn.RMSNorm(embed_dim)

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

model = MiniLLaMAMoE(
    vocab_size   = tokenizer.n_vocab,
    embed_dim    = 32,
    num_heads    = 4,
    num_kv_heads = 2,    # 2 KV heads — each shared by 2 Q heads
    num_layers   = 2,
    num_experts  = 8,    # 8 experts per layer — same as LLaMA 4 Scout
    top_k        = 2     # each token routed to 2 experts
)

print(model)

# Print parameter breakdown
total_params  = sum(p.numel() for p in model.parameters())
active_params = sum(p.numel() for p in model.blocks[0].moe.experts[0].parameters()) * 2
print(f"\nTotal parameters:         {total_params:,}")
print(f"Active MoE params/token:  {active_params:,} ({100 * active_params / total_params:.1f}% of total)")

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

print(f"\nTrain sequences:      {len(train_dataset)}")
print(f"Validation sequences: {len(val_dataset)}")
print(f"Batches per epoch:    {len(train_loader)}")

# AdamW — Adam with weight decay, standard for transformer training
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

model.train()
step = 0

# outer loop keeps cycling through the dataloader until max_steps is reached
while step < max_steps:
    for x, y in train_loader:
        if step >= max_steps:
            break

        x = x.to(device)
        y = y.to(device)

        logits  = model(x)
        B, T, C = logits.shape

        # Cross entropy loss — measures how well the model predicts the next token
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