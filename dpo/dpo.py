import copy
import json
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

# ── Default — tiny_corpus.txt + sft_dataset.jsonl + dpo_dataset.jsonl ──
# Sized to match the tiny corpus (~700 words, 20 SFT examples, 15 DPO examples)
# A larger model would immediately overfit this data
# Swap to the curated presets below when using fineweb/dolma corpora
embed_dim          = 32
num_heads          = 4
hidden_dim         = 128
num_layers         = 2
block_size         = 32
pretrain_max_steps = 5000
sft_max_steps      = 1000
dpo_max_steps      = 500

# ── Curated corpus (fineweb_corpus.txt + fineweb_sft.jsonl + fineweb_dpo.jsonl) ──
# Use these when swapping to a larger curated corpus from data_curation/
# max_steps must scale with corpus — 5000 pretrain steps barely touches a large corpus
# ── Laptop / GPU < 40GB VRAM ──
# embed_dim          = 256
# num_heads          = 8
# hidden_dim         = 1024
# num_layers         = 8
# block_size         = 128
# pretrain_max_steps = 50000
# sft_max_steps      = 5000
# dpo_max_steps      = 1000

# ── Cloud GPU ≥ 40GB VRAM (A100, H100) ──
# embed_dim          = 512
# num_heads          = 16
# hidden_dim         = 2048
# num_layers         = 12
# block_size         = 256
# pretrain_max_steps = 200000
# sft_max_steps      = 20000
# dpo_max_steps      = 5000
# USE_COMPILE        = True    # torch.compile — significant speedup on CUDA
# USE_AMP            = True    # automatic mixed precision — CUDA only, not MPS

# ── CPU / Small GPU (Tesla V100) ──
# embed_dim          = 128
# num_heads          = 4
# hidden_dim         = 512
# num_layers         = 4
# block_size         = 64
# pretrain_max_steps = 20000
# sft_max_steps      = 2000
# dpo_max_steps      = 500


# ─── Config ───────────────────────────────────────────────────────────────────

# ── Default — tiny_corpus.txt ──
CORPUS_PATH   = "../data/tiny_corpus.txt"
SFT_DATA_PATH = "../data/sft_dataset.jsonl"
DPO_DATA_PATH = "../data/dpo_dataset.jsonl"
MODEL_SAVE    = "../data/dpo_model.pt"

# ── FineWeb curated corpus ──
# CORPUS_PATH   = "../data/fineweb_corpus.txt"
# SFT_DATA_PATH = "../data/fineweb_sft.jsonl"
# DPO_DATA_PATH = "../data/fineweb_dpo.jsonl"
# MODEL_SAVE    = "../data/fineweb_dpo_model.pt"

# ── Dolma curated corpus ──
# CORPUS_PATH   = "../data/dolma_corpus.txt"
# SFT_DATA_PATH = "../data/dolma_sft.jsonl"
# DPO_DATA_PATH = "../data/dolma_dpo.jsonl"
# MODEL_SAVE    = "../data/dolma_dpo_model.pt"

# ── Mixed curated corpus ──
# CORPUS_PATH   = "../data/mixed_corpus.txt"
# SFT_DATA_PATH = "../data/mixed_sft.jsonl"
# DPO_DATA_PATH = "../data/mixed_dpo.jsonl"
# MODEL_SAVE    = "../data/mixed_dpo_model.pt"

batch_size    = 4   # small — SFT and DPO datasets are tiny


# ─── Tokenizer ────────────────────────────────────────────────────────────────

tokenizer = tiktoken.encoding_for_model("gpt-4o")


# ─── Pretraining Dataset ──────────────────────────────────────────────────────

class PretrainDataset(Dataset):
    def __init__(self, path, tokenizer, block_size):
        with open(path, "r", encoding="utf8") as f:
            text = f.read()

        data            = tokenizer.encode(text)
        self.data       = torch.tensor(data, dtype=torch.long)
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.block_size]
        y = self.data[idx + 1:idx + self.block_size + 1]
        return x, y


# ─── SFT Dataset ──────────────────────────────────────────────────────────────

class SFTDataset(Dataset):
    def __init__(self, path, tokenizer, block_size):
        self.examples   = []
        self.block_size = block_size

        with open(path, "r", encoding="utf8") as f:
            for line in f:
                example = json.loads(line.strip())

                text = (
                    f"### Instruction:\n{example['instruction']}\n\n"
                    f"### Response:\n{example['response']}"
                )

                tokens = tokenizer.encode(text)

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


# ─── DPO Dataset ──────────────────────────────────────────────────────────────

class DPODataset(Dataset):
    def __init__(self, path, tokenizer, block_size):
        self.examples   = []
        self.block_size = block_size

        with open(path, "r", encoding="utf8") as f:
            for line in f:
                example = json.loads(line.strip())

                chosen_text = (
                    f"### Prompt:\n{example['prompt']}\n\n"
                    f"### Response:\n{example['chosen']}"
                )
                rejected_text = (
                    f"### Prompt:\n{example['prompt']}\n\n"
                    f"### Response:\n{example['rejected']}"
                )

                chosen_tokens   = tokenizer.encode(chosen_text)
                rejected_tokens = tokenizer.encode(rejected_text)

                def pad(tokens):
                    if len(tokens) < block_size:
                        return tokens + [0] * (block_size - len(tokens))
                    return tokens[:block_size]

                self.examples.append({
                    "chosen":   torch.tensor(pad(chosen_tokens),   dtype=torch.long),
                    "rejected": torch.tensor(pad(rejected_tokens), dtype=torch.long)
                })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


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


# ─── Stage 1 — Pretraining ────────────────────────────────────────────────────

def pretrain(model, device):
    # Train on raw text — builds the base language understanding
    # the model learns token distributions, grammar, and world knowledge
    # from the corpus before any instruction tuning
    print("=" * 60)
    print("Stage 1 — Pretraining on corpus")
    print("=" * 60)

    dataset    = PretrainDataset(CORPUS_PATH, tokenizer, block_size)
    split_idx  = int(0.9 * len(dataset))
    train_data = torch.utils.data.Subset(dataset, range(0, split_idx))
    loader     = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    optimizer  = optim.AdamW(model.parameters(), lr=3e-4)

    model.train()
    step = 0

    while step < pretrain_max_steps:
        for x, y in loader:
            if step >= pretrain_max_steps:
                break

            x, y    = x.to(device), y.to(device)
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


# ─── Stage 2 — SFT ────────────────────────────────────────────────────────────

def sft_train(model, device):
    # Supervised fine tuning on instruction/response pairs
    # teaches the model to follow instructions using signals
    # already present from pretraining
    print("=" * 60)
    print("Stage 2 — SFT on sft dataset")
    print("=" * 60)

    dataset   = SFTDataset(SFT_DATA_PATH, tokenizer, block_size)
    loader    = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Freeze embeddings — preserve token representations from pretraining
    for param in model.token_embedding.parameters():
        param.requires_grad = False
    for param in model.position_embedding.parameters():
        param.requires_grad = False

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4   # lower than pretraining — nudging not rewriting
    )

    print(f"SFT dataset size: {len(dataset)} examples")

    model.train()
    step = 0

    while step < sft_max_steps:
        for x, y in loader:
            if step >= sft_max_steps:
                break

            x, y    = x.to(device), y.to(device)
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

    # Unfreeze embeddings after SFT
    for param in model.token_embedding.parameters():
        param.requires_grad = True
    for param in model.position_embedding.parameters():
        param.requires_grad = True

    print("SFT complete\n")


# ─── Stage 3 — DPO ────────────────────────────────────────────────────────────

def compute_log_probs(model, tokens):
    x      = tokens[:, :-1]
    y      = tokens[:, 1:]
    logits = model(x)
    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(2, y.unsqueeze(-1)).squeeze(-1)
    return token_log_probs.sum(dim=-1)


def dpo_loss(model, ref_model, chosen, rejected, beta=0.1):
    # DPO loss — Direct Preference Optimization
    # trains the model to prefer chosen responses over rejected ones
    # without needing a separate reward model (unlike RLHF)
    chosen_log_probs   = compute_log_probs(model, chosen)
    rejected_log_probs = compute_log_probs(model, rejected)

    with torch.no_grad():
        ref_chosen_log_probs   = compute_log_probs(ref_model, chosen)
        ref_rejected_log_probs = compute_log_probs(ref_model, rejected)

    chosen_ratio   = chosen_log_probs   - ref_chosen_log_probs
    rejected_ratio = rejected_log_probs - ref_rejected_log_probs

    loss = -F.logsigmoid(beta * (chosen_ratio - rejected_ratio)).mean()
    return loss


def dpo_train(model, ref_model, device):
    # Align the SFT model using preference data
    # DPO directly optimizes the model to prefer chosen over rejected responses
    # no reward model needed — the preference signal comes from the dataset
    print("=" * 60)
    print("Stage 3 — DPO on dpo dataset")
    print("=" * 60)

    dataset   = DPODataset(DPO_DATA_PATH, tokenizer, block_size)
    loader    = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.AdamW(model.parameters(), lr=5e-5)  # very low lr for alignment

    print(f"DPO dataset size: {len(dataset)} examples")

    model.train()
    ref_model.eval()  # reference model is always frozen
    step = 0

    while step < dpo_max_steps:
        for batch in loader:
            if step >= dpo_max_steps:
                break

            chosen   = batch["chosen"].to(device)
            rejected = batch["rejected"].to(device)

            loss = dpo_loss(model, ref_model, chosen, rejected)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 50 == 0:
                print(f"  DPO step {step}, Loss: {loss.item():.4f}")

            step += 1

    print("DPO complete\n")


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

    model = MiniGPT(
        vocab_size  = tokenizer.n_vocab,
        block_size  = block_size,
        embed_dim   = embed_dim,
        num_heads   = num_heads,
        hidden_dim  = hidden_dim,
        num_layers  = num_layers
    ).to(device)

    # ── Stage 1 — Pretrain ────────────────────────────────────────────────────
    pretrain(model, device)

    print("After pretraining:")
    print(generate(model, "The lighthouse had been dark"))
    print()

    # ── Stage 2 — SFT ─────────────────────────────────────────────────────────
    sft_train(model, device)

    print("After SFT:")
    print(generate(model, "### Instruction:\nWho is Maria?\n\n### Response:\n"))
    print()

    # ── Stage 3 — DPO ─────────────────────────────────────────────────────────

    # Create a frozen copy of the SFT model to use as the reference
    # the reference model captures the SFT distribution before DPO nudges it
    ref_model = copy.deepcopy(model).to(device)
    for param in ref_model.parameters():
        param.requires_grad = False

    dpo_train(model, ref_model, device)

    print("After DPO:")
    print(generate(model, "### Prompt:\nWhy did Maria move to the island?\n\n### Response:\n"))
    print()

    # ── Save ──────────────────────────────────────────────────────────────────
    torch.save(model.state_dict(), MODEL_SAVE)
    print(f"Model saved to {MODEL_SAVE}")