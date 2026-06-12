import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken
from torch.utils.data import Dataset, DataLoader


# ─── Config ───────────────────────────────────────────────────────────────────

CORPUS_PATH   = "../data/tiny_corpus.txt"
SFT_DATA_PATH = "../data/sft_dataset.jsonl"
DPO_DATA_PATH = "../data/dpo_dataset.jsonl"
MODEL_SAVE    = "../data/dpo_model.pt"

block_size    = 32
embed_dim     = 32
num_heads     = 4
hidden_dim    = 128
num_layers    = 2
batch_size    = 4


# ─── Tokenizer ────────────────────────────────────────────────────────────────

tokenizer = tiktoken.encoding_for_model("gpt-4o")


# ─── Pretraining Dataset ──────────────────────────────────────────────────────

class PretrainDataset(Dataset):
    def __init__(self, path, tokenizer, block_size):
        with open(path, "r", encoding="utf8") as f:
            text = f.read()

        # Tokenize the entire corpus into a flat list of integer token IDs
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

                # Format as instruction/response pair
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


# ─── DPO Dataset ──────────────────────────────────────────────────────────────

class DPODataset(Dataset):
    def __init__(self, path, tokenizer, block_size):
        self.examples   = []
        self.block_size = block_size

        with open(path, "r", encoding="utf8") as f:
            for line in f:
                example = json.loads(line.strip())

                # Format chosen and rejected responses as full prompt/response pairs
                # DPO needs both to compute the preference loss
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

                # Pad or truncate both to block_size
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
    print("Stage 1 — Pretraining on tiny_corpus.txt")
    print("=" * 60)

    dataset    = PretrainDataset(CORPUS_PATH, tokenizer, block_size)
    split_idx  = int(0.9 * len(dataset))
    train_data = torch.utils.data.Subset(dataset, range(0, split_idx))
    loader     = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    optimizer  = torch.optim.AdamW(model.parameters(), lr=3e-4)

    model.train()
    step = 0

    while step < 5000:
        for x, y in loader:
            if step >= 5000:
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
    # fine tuning a model with no pretraining signal produces poor results
    print("=" * 60)
    print("Stage 2 — SFT on sft_dataset.jsonl")
    print("=" * 60)

    dataset   = SFTDataset(SFT_DATA_PATH, tokenizer, block_size)
    loader    = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Freeze embeddings — preserve token representations from pretraining
    # only update the transformer blocks during fine tuning
    for param in model.token_embedding.parameters():
        param.requires_grad = False
    for param in model.position_embedding.parameters():
        param.requires_grad = False

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4   # lower than pretraining — nudging not rewriting
    )

    print(f"SFT dataset size: {len(dataset)} examples")

    model.train()
    step = 0

    while step < 1000:
        for x, y in loader:
            if step >= 1000:
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
    # Compute the log probability of a sequence under the model
    # used to measure how much the model prefers chosen vs rejected responses
    x      = tokens[:, :-1]
    y      = tokens[:, 1:]
    logits = model(x)
    log_probs = F.log_softmax(logits, dim=-1)

    # Gather the log prob of each actual token
    # shape: (batch, seq_len)
    token_log_probs = log_probs.gather(
        2, y.unsqueeze(-1)
    ).squeeze(-1)

    # Sum log probs across the sequence — overall sequence log probability
    return token_log_probs.sum(dim=-1)


def dpo_loss(model, ref_model, chosen, rejected, beta=0.1):
    # DPO loss — Direct Preference Optimization
    # trains the model to prefer chosen responses over rejected ones
    # without needing a separate reward model (unlike RLHF)
    #
    # beta controls the strength of the KL penalty
    # low beta  — model diverges more freely from the reference
    # high beta — model stays closer to the reference (safer but less aligned)
    #
    # the reference model is a frozen copy of the SFT model
    # it acts as a baseline — DPO pushes the policy model toward chosen
    # while the KL term prevents it from drifting too far from the reference

    # Policy model log probs
    chosen_log_probs   = compute_log_probs(model, chosen)
    rejected_log_probs = compute_log_probs(model, rejected)

    # Reference model log probs — no gradients needed
    with torch.no_grad():
        ref_chosen_log_probs   = compute_log_probs(ref_model, chosen)
        ref_rejected_log_probs = compute_log_probs(ref_model, rejected)

    # Log ratio of policy to reference for chosen and rejected
    # positive ratio means policy prefers this response more than reference did
    chosen_ratio   = chosen_log_probs   - ref_chosen_log_probs
    rejected_ratio = rejected_log_probs - ref_rejected_log_probs

    # DPO loss — maximize the margin between chosen and rejected ratios
    # sigmoid ensures the loss is bounded and stable
    loss = -F.logsigmoid(beta * (chosen_ratio - rejected_ratio)).mean()

    return loss


def dpo_train(model, ref_model, device):
    # Align the SFT model using preference data
    # DPO directly optimizes the model to prefer chosen over rejected responses
    # no reward model needed — the preference signal comes from the dataset
    print("=" * 60)
    print("Stage 3 — DPO on dpo_dataset.jsonl")
    print("=" * 60)

    dataset   = DPODataset(DPO_DATA_PATH, tokenizer, block_size)
    loader    = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)  # very low lr for alignment

    print(f"DPO dataset size: {len(dataset)} examples")

    model.train()
    ref_model.eval()  # reference model is always frozen
    step = 0

    while step < 500:
        for batch in loader:
            if step >= 500:
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
    device = next(model.parameters()).device
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
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build model
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

    # Test after pretraining — raw completion, no instruction following
    print("After pretraining:")
    print(generate(model, "The lighthouse had been dark"))
    print()

    # ── Stage 2 — SFT ─────────────────────────────────────────────────────────
    sft_train(model, device)

    # Test after SFT — should follow the instruction format
    print("After SFT:")
    print(generate(model, "### Instruction:\nWho is Maria?\n\n### Response:\n"))
    print()

    # ── Stage 3 — DPO ─────────────────────────────────────────────────────────

    # Create a frozen copy of the SFT model to use as the reference
    # the reference model captures the SFT distribution before DPO nudges it
    import copy
    ref_model = copy.deepcopy(model).to(device)
    for param in ref_model.parameters():
        param.requires_grad = False

    dpo_train(model, ref_model, device)

    # Test after DPO — should prefer accurate responses over vague ones
    print("After DPO:")
    print(generate(model, "### Prompt:\nWhy did Maria move to the island?\n\n### Response:\n"))
    print()

    # ── Save ──────────────────────────────────────────────────────────────────
    torch.save(model.state_dict(), MODEL_SAVE)
    print(f"Model saved to {MODEL_SAVE}")