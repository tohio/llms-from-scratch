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


# ─── Greedy Generation ────────────────────────────────────────────────────────

def generate_greedy(model, tokenizer, prompt, block_size, max_new_tokens=50):
    model.eval()
    device = next(model.parameters()).device
    tokens = tokenizer.encode(prompt)
    x      = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            x_cond = x[:, -block_size:]
            logits = model(x_cond)

            # Take the last token position
            logits = logits[:, -1, :]

            # Greedy — always pick the single highest probability token
            # deterministic — same prompt always produces the same output
            # tends to produce repetitive or generic text
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
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
    print(f"Greedy (15 tokens):  {generate_greedy(model, tokenizer, prompt, block_size, 15)}")
    print(f"Greedy (50 tokens):  {generate_greedy(model, tokenizer, prompt, block_size, 50)}")