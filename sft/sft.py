import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken
from torch.utils.data import Dataset, DataLoader


# ─── Hardware Config ──────────────────────────────────────────────────────────

device = torch.device(
    "mps"  if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available()         else
    "cpu"
)

print(f"Using device: {device}")

# ── Default — tiny_corpus.txt + sft_dataset.jsonl ──
# Sized to match the tiny corpus (~700 words, 20 SFT examples)
# A larger model would immediately overfit this data
# Swap to the curated presets below when using fineweb/dolma corpora
embed_dim  = 32
num_heads  = 4
hidden_dim = 128
num_layers = 2
block_size = 32

# ── Curated corpus (fineweb_corpus.txt + fineweb_sft.jsonl) ──
# Use these when swapping to a larger curated corpus from data_curation/
# ── Laptop / M4 Max ──
# embed_dim  = 256
# num_heads  = 8
# hidden_dim = 1024
# num_layers = 8
# block_size = 128

# ── Cloud GPU (A100/H100) ──
# embed_dim    = 512
# num_heads    = 16
# hidden_dim   = 2048
# num_layers   = 12
# block_size   = 256
# USE_COMPILE  = True    # torch.compile — significant speedup on CUDA
# USE_AMP      = True    # automatic mixed precision — CUDA only, not MPS

# ── CPU / Small GPU ──
# embed_dim  = 128
# num_heads  = 4
# hidden_dim = 512
# num_layers = 4
# block_size = 64


# ─── Config ───────────────────────────────────────────────────────────────────

# Swap these paths when using curated corpora from data_curation/
# e.g. CORPUS_PATH = "../data/fineweb_corpus.txt"
#      SFT_DATA_PATH = "../data/fineweb_sft.jsonl"
CORPUS_PATH   = "../data/tiny_corpus.txt"
SFT_DATA_PATH = "../data/sft_dataset.jsonl"
MODEL_SAVE    = "../data/sft_model.pt"

batch_size    = 4      # small — SFT dataset is tiny
max_steps     = 1000   # SFT needs fewer steps than pretraining
learning_rate = 1e-4   # lower than pretraining — fine tuning not retraining


# ─── Tokenizer ────────────────────────────────────────────────────────────────

tokenizer = tiktoken.encoding_for_model("gpt-4o")


# ─── SFT Dataset ──────────────────────────────────────────────────────────────

class SFTDataset(Dataset):
    def __init__(self, path, tokenizer, block_size):
        self.examples   = []
        self.block_size = block_size

        with open(path, "r", encoding="utf8") as f:
            for line in f:
                example = json.loads(line.strip())

                # Format as a prompt/response pair
                # the model learns to complete the response given the instruction
                # special tokens mark the boundary between instruction and response
                text = (
                    f"### Instruction:\n{example['instruction']}\n\n"
                    f"### Response:\n{example['response']}"
                )

                tokens = tokenizer.encode(text)

                # Pad or truncate to block_size
                if len(tokens) < block_size:
                    tokens = tokens + [0] * (block_size - len(tokens))
                else:
                    tokens = tokens[:block_size]

                self.examples.append(tokens)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        tokens = self.examples[idx]
        x      = torch.tensor(tokens[:-1], dtype=torch.long)
        y      = torch.tensor(tokens[1:],  dtype=torch.long)
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


# ─── Pretrain on Corpus ───────────────────────────────────────────────────────

def pretrain(model, device):
    # Pretrain on corpus first so the model has signal before SFT
    # fine tuning a model with no pretraining signal produces poor results
    # regardless of SFT data quality — the signal must already be latent
    print("Pretraining on corpus...")

    with open(CORPUS_PATH, "r", encoding="utf8") as f:
        text = f.read()

    data      = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    model.train()
    step = 0

    while step < 5000:
        indices = torch.randint(0, len(data) - block_size, (batch_size,))
        x       = torch.stack([data[i:i + block_size] for i in indices]).to(device)
        y       = torch.stack([data[i + 1:i + block_size + 1] for i in indices]).to(device)

        logits  = model(x)
        B, T, C = logits.shape
        loss    = F.cross_entropy(logits.view(B * T, C), y.view(B * T))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 500 == 0:
            print(f"  Pretrain step {step}, Loss: {loss.item():.4f}")

        step += 1

    print("Pretraining complete\n")


# ─── SFT Training ─────────────────────────────────────────────────────────────

def sft_train(model, dataloader, device):
    # Fine tune the pretrained model on instruction/response pairs
    # lower learning rate than pretraining — we want to nudge not overwrite
    print("Starting SFT...")

    # Freeze the embedding layers — preserve the token representations
    # learned during pretraining, only update the transformer blocks
    for param in model.token_embedding.parameters():
        param.requires_grad = False
    for param in model.position_embedding.parameters():
        param.requires_grad = False

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate
    )

    model.train()
    step = 0

    while step < max_steps:
        for x, y in dataloader:
            if step >= max_steps:
                break

            x = x.to(device)
            y = y.to(device)

            logits  = model(x)
            B, T, C = logits.shape

            # Standard cross entropy loss on the full sequence
            # in production SFT you would mask the instruction tokens
            # and only compute loss on the response tokens
            # here we keep it simple and train on the full sequence
            loss = F.cross_entropy(logits.view(B * T, C), y.view(B * T))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 100 == 0:
                print(f"  SFT step {step}, Loss: {loss.item():.4f}")

            step += 1

    print("SFT complete\n")


# ─── Generation ───────────────────────────────────────────────────────────────

def generate(model, prompt, max_new_tokens=100):
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


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(123)

    # Build model
    model = MiniGPT(
        vocab_size  = tokenizer.n_vocab,
        block_size  = block_size,
        embed_dim   = embed_dim,
        num_heads   = num_heads,
        hidden_dim  = hidden_dim,
        num_layers  = num_layers
    ).to(device)

    # Step 1 — pretrain on corpus
    pretrain(model, device)

    # Step 2 — fine tune on SFT dataset
    sft_dataset = SFTDataset(SFT_DATA_PATH, tokenizer, block_size)
    sft_loader  = DataLoader(sft_dataset, batch_size=batch_size, shuffle=True)

    print(f"SFT dataset size: {len(sft_dataset)} examples")
    sft_train(model, sft_loader, device)

    # Step 3 — save the fine tuned model
    torch.save(model.state_dict(), MODEL_SAVE)
    print(f"Model saved to {MODEL_SAVE}")

    # Step 4 — test with instruction prompts
    prompt = "### Instruction:\nWho is Maria?\n\n### Response:\n"
    print(f"\nPrompt: {prompt}")
    print(f"Response: {generate(model, prompt)}")

    prompt = "### Instruction:\nWhat happened during the storm?\n\n### Response:\n"
    print(f"\nPrompt: {prompt}")
    print(f"Response: {generate(model, prompt)}")