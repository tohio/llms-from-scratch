import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken
from torch.utils.data import Dataset, DataLoader


# ─── Dataset ──────────────────────────────────────────────────────────────────

class TinyCorpusDataset(Dataset):
    def __init__(self, path, block_size):
        with open(path, "r", encoding="utf8") as f:
            text = f.read()
        tokenizer       = tiktoken.encoding_for_model("gpt-4o")
        data            = tokenizer.encode(text)
        self.data       = torch.tensor(data, dtype=torch.long)
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.block_size]
        y = self.data[idx + 1:idx + self.block_size + 1]
        return x, y


# ─── Model ────────────────────────────────────────────────────────────────────

class MaskedMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model   = d_model
        self.num_heads = num_heads
        self.head_dim  = d_model // num_heads
        self.q_proj    = nn.Linear(d_model, d_model, bias=False)
        self.k_proj    = nn.Linear(d_model, d_model, bias=False)
        self.v_proj    = nn.Linear(d_model, d_model, bias=False)
        self.out_proj  = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        b, seq_len, _ = x.shape
        Q = self.q_proj(x).view(b, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(b, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(b, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        scores = Q @ K.transpose(-2, -1) / (self.head_dim ** 0.5)
        mask   = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(mask, float('-inf'))
        attn_weights = torch.softmax(scores, dim=-1)
        context = (attn_weights @ V).transpose(1, 2).contiguous().view(b, seq_len, self.d_model)
        return self.out_proj(context)


class FeedForward(nn.Module):
    def __init__(self, d_model, hidden_dim):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
        )

    def forward(self, x):
        return self.ffn(x)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim):
        super().__init__()
        self.attention = MaskedMultiHeadAttention(embed_dim, num_heads)
        self.ffn       = FeedForward(embed_dim, hidden_dim)
        self.norm1     = nn.LayerNorm(embed_dim)
        self.norm2     = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class MiniGPT(nn.Module):
    def __init__(self, vocab_size, block_size, embed_dim, num_heads, hidden_dim, num_layers):
        super().__init__()
        self.token_embedding    = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(block_size, embed_dim)
        self.blocks             = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads, hidden_dim)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(embed_dim)
        self.lm_head    = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        b, seq_len          = x.shape
        token_embeddings    = self.token_embedding(x)
        positions           = torch.arange(seq_len, device=x.device)
        position_embeddings = self.position_embedding(positions)
        x = token_embeddings + position_embeddings
        x = self.blocks(x)
        x = self.final_norm(x)
        return self.lm_head(x)


# ─── Train ────────────────────────────────────────────────────────────────────

def train(model, dataset, device, batch_size=32, max_steps=5000, lr=3e-4):
    split_idx     = int(0.9 * len(dataset))
    train_dataset = torch.utils.data.Subset(dataset, range(0, split_idx))
    train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    optimizer     = torch.optim.AdamW(model.parameters(), lr=lr)

    model.train()
    for step, (x, y) in enumerate(train_loader):
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


# ─── Top-P Generation ─────────────────────────────────────────────────────────

def generate_top_p(model, tokenizer, prompt, block_size, max_new_tokens=50, p=0.9):
    model.eval()
    device = next(model.parameters()).device
    tokens = tokenizer.encode(prompt)
    x      = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            x_cond = x[:, -block_size:]
            logits = model(x_cond)
            logits = logits[:, -1, :]

            # Convert logits to probabilities
            probs = torch.softmax(logits, dim=-1)

            # Sort probabilities in descending order
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)

            # Compute cumulative probabilities
            # cumsum gives a running total — e.g. [0.3, 0.5, 0.7, 0.85, ...]
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

            # Remove tokens once cumulative probability exceeds p
            # shift by one position so we always keep at least one token
            # tokens beyond the nucleus get zeroed out
            sorted_indices_to_remove = cumulative_probs - sorted_probs > p
            sorted_probs[sorted_indices_to_remove] = 0.0

            # Renormalise the remaining probabilities so they sum to 1
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

            # Sample from the nucleus
            sampled_index = torch.multinomial(sorted_probs, num_samples=1)

            # Map back to the original vocabulary indices
            next_token = sorted_indices.gather(1, sampled_index)
            x          = torch.cat([x, next_token], dim=1)

    return tokenizer.decode(x[0].tolist())


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    block_size = 4
    tokenizer  = tiktoken.encoding_for_model("gpt-4o")
    device     = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(123)
    dataset = TinyCorpusDataset("../data/tiny_corpus.txt", block_size=block_size)
    model   = MiniGPT(
        vocab_size  = tokenizer.n_vocab,
        block_size  = block_size,
        embed_dim   = 32,
        num_heads   = 4,
        hidden_dim  = 128,
        num_layers  = 2
    ).to(device)

    print("Training...")
    train(model, dataset, device)

    prompt = "The lighthouse"
    print(f"\nPrompt: {prompt}")

    # Compare different P values
    # p=0.5 — tight nucleus, only the most likely tokens
    # p=0.9 — standard setting used in most production models
    # p=1.0 — no filtering, all tokens are candidates
    for p in [0.5, 0.9, 0.95, 1.0]:
        output = generate_top_p(model, tokenizer, prompt, block_size, 50, p=p)
        print(f"Top-P (p={p}): {output}")