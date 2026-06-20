import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken
import urllib.request
import os
from torch.utils.data import Dataset, DataLoader


# ─── Hardware Config ──────────────────────────────────────────────────────────

device = (
    "mps"  if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available()         else
    "cpu"
)

print(f"Using device: {device}")

# ── Laptop / M4 Max ──
TEACHER_EMBED_DIM  = 256
TEACHER_NUM_HEADS  = 8
TEACHER_HIDDEN_DIM = 1024
TEACHER_NUM_LAYERS = 8

STUDENT_EMBED_DIM  = 128
STUDENT_NUM_HEADS  = 4
STUDENT_HIDDEN_DIM = 512
STUDENT_NUM_LAYERS = 4

block_size    = 128
TEACHER_STEPS = 10000
STUDENT_STEPS = 5000

# ── Cloud GPU (A100/H100) ──
# TEACHER_EMBED_DIM  = 512
# TEACHER_NUM_HEADS  = 16
# TEACHER_HIDDEN_DIM = 2048
# TEACHER_NUM_LAYERS = 12
# STUDENT_EMBED_DIM  = 256
# STUDENT_NUM_HEADS  = 8
# STUDENT_HIDDEN_DIM = 1024
# STUDENT_NUM_LAYERS = 6
# block_size         = 256
# TEACHER_STEPS      = 50000
# STUDENT_STEPS      = 25000
# USE_COMPILE        = True
# USE_AMP            = True

# ── CPU / Small GPU (Tesla V100) ──
# TEACHER_EMBED_DIM  = 128
# TEACHER_NUM_HEADS  = 4
# TEACHER_HIDDEN_DIM = 512
# TEACHER_NUM_LAYERS = 4
# STUDENT_EMBED_DIM  = 64
# STUDENT_NUM_HEADS  = 2
# STUDENT_HIDDEN_DIM = 256
# STUDENT_NUM_LAYERS = 2
# block_size         = 64
# TEACHER_STEPS      = 10000
# STUDENT_STEPS      = 5000


# ─── Config ───────────────────────────────────────────────────────────────────

CORPUS_URLS = {
    "sherlock": "https://www.gutenberg.org/files/1661/1661-0.txt",
    "darwin":   "https://www.gutenberg.org/files/1228/1228-0.txt"
}
CORPUS_PATHS = {
    "sherlock": "../data/sherlock_corpus.txt",
    "darwin":   "../data/darwin_corpus.txt"
}
COMBINED_CORPUS_PATH = "../data/reasoning_corpus.txt"
TEACHER_SAVE         = "../data/distillation_teacher.pt"
STUDENT_SAVE         = "../data/distillation_features_student.pt"

BATCH_SIZE  = 16
TEACHER_LR  = 3e-4
STUDENT_LR  = 1e-4

# Feature distillation hyperparameters
ALPHA = 0.5   # weight for hard label loss (cross entropy)
BETA  = 0.5   # weight for feature matching loss (MSE on hidden states)


# ─── Tokenizer ────────────────────────────────────────────────────────────────

tokenizer = tiktoken.encoding_for_model("gpt-4o")


# ─── Corpus ───────────────────────────────────────────────────────────────────

def strip_gutenberg(text):
    start = text.find("*** START OF")
    end   = text.find("*** END OF")
    if start != -1 and end != -1:
        text = text[start:end]
    return text.strip()


def download_corpus():
    combined = []

    for name, url in CORPUS_URLS.items():
        path = CORPUS_PATHS[name]

        if not os.path.exists(path):
            print(f"Downloading {name} corpus...")
            urllib.request.urlretrieve(url, path)
            print(f"Saved to {path}")
        else:
            print(f"{name} corpus already exists at {path}")

        with open(path, "r", encoding="utf8") as f:
            text = f.read()

        text = strip_gutenberg(text)
        combined.append(text)
        print(f"{name}: {len(text):,} characters")

    if os.path.exists(COMBINED_CORPUS_PATH):
        print(f"\nCombined corpus already exists at {COMBINED_CORPUS_PATH}")
        with open(COMBINED_CORPUS_PATH, "r", encoding="utf8") as f:
            return f.read()

    full_text = "\n\n---\n\n".join(combined)

    with open(COMBINED_CORPUS_PATH, "w", encoding="utf8") as f:
        f.write(full_text)

    print(f"\nCombined corpus: {len(full_text):,} characters")
    print(f"Saved to {COMBINED_CORPUS_PATH}")
    return full_text


# ─── Dataset ──────────────────────────────────────────────────────────────────

class TextDataset(Dataset):
    def __init__(self, data, block_size):
        self.data       = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.block_size]
        y = self.data[idx + 1:idx + self.block_size + 1]
        return x, y


# ─── Model ────────────────────────────────────────────────────────────────────
# forward() returns both logits AND the final hidden state
# feature distillation needs access to intermediate representations

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


class GPT(nn.Module):
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

    def forward(self, x, return_hidden=False):
        b, seq_len          = x.shape
        token_embeddings    = self.token_embedding(x)
        positions           = torch.arange(seq_len, device=x.device)
        position_embeddings = self.position_embedding(positions)
        x = token_embeddings + position_embeddings
        x = self.blocks(x)
        hidden = self.final_norm(x)
        logits = self.lm_head(hidden)

        if return_hidden:
            return logits, hidden
        return logits


# ─── Projection Layer ─────────────────────────────────────────────────────────

class FeatureProjection(nn.Module):
    # Teacher and student have different embed_dim — hidden states live in
    # different vector spaces and cannot be compared directly.
    # This projection maps student hidden states into the teacher's space.
    def __init__(self, student_dim, teacher_dim):
        super().__init__()
        self.proj = nn.Linear(student_dim, teacher_dim, bias=False)

    def forward(self, x):
        return self.proj(x)


# ─── Train Teacher ────────────────────────────────────────────────────────────

def train_teacher(model, data, device):
    print("\n" + "=" * 60)
    print("Stage 1 — Training Teacher")
    print("=" * 60)

    dataset      = TextDataset(data, block_size)
    split_idx    = int(0.9 * len(dataset))
    train_data   = torch.utils.data.Subset(dataset, range(0, split_idx))
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    optimizer    = torch.optim.AdamW(model.parameters(), lr=TEACHER_LR)

    model.train()
    step = 0

    while step < TEACHER_STEPS:
        for x, y in train_loader:
            if step >= TEACHER_STEPS:
                break

            x, y    = x.to(device), y.to(device)
            logits  = model(x)
            B, T, C = logits.shape
            loss    = F.cross_entropy(logits.view(B * T, C), y.view(B * T))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 500 == 0:
                print(f"  Teacher step {step}, Loss: {loss.item():.4f}")

            step += 1

    torch.save(model.state_dict(), TEACHER_SAVE)
    print(f"Teacher saved to {TEACHER_SAVE}\n")


# ─── Feature Distillation Loss ────────────────────────────────────────────────

def feature_distillation_loss(student_logits, student_hidden, teacher_hidden,
                               projection, targets, alpha, beta):
    B, T, C = student_logits.shape

    hard_loss = F.cross_entropy(
        student_logits.view(B * T, C),
        targets.view(B * T)
    )

    projected_student_hidden = projection(student_hidden)

    feature_loss = F.mse_loss(
        projected_student_hidden,
        teacher_hidden.detach()
    )

    return alpha * hard_loss + beta * feature_loss, hard_loss, feature_loss


# ─── Train Student ────────────────────────────────────────────────────────────

def train_student(student, teacher, projection, data, device):
    print("\n" + "=" * 60)
    print("Stage 2 — Feature Distillation")
    print("=" * 60)
    print(f"Alpha (hard):    {ALPHA}")
    print(f"Beta  (feature): {BETA}\n")

    dataset      = TextDataset(data, block_size)
    split_idx    = int(0.9 * len(dataset))
    train_data   = torch.utils.data.Subset(dataset, range(0, split_idx))
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = torch.optim.AdamW(
        list(student.parameters()) + list(projection.parameters()),
        lr=STUDENT_LR
    )

    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False

    student.train()
    projection.train()
    step = 0

    while step < STUDENT_STEPS:
        for x, y in train_loader:
            if step >= STUDENT_STEPS:
                break

            x, y = x.to(device), y.to(device)

            student_logits, student_hidden = student(x, return_hidden=True)

            with torch.no_grad():
                _, teacher_hidden = teacher(x, return_hidden=True)

            loss, hard_loss, feature_loss = feature_distillation_loss(
                student_logits, student_hidden, teacher_hidden,
                projection, y, ALPHA, BETA
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 500 == 0:
                print(f"  Step {step} — Total: {loss.item():.4f} | Hard: {hard_loss.item():.4f} | Feature: {feature_loss.item():.4f}")

            step += 1

    torch.save(student.state_dict(), STUDENT_SAVE)
    print(f"\nStudent saved to {STUDENT_SAVE}")


# ─── Baseline Student (no distillation) ──────────────────────────────────────

def train_baseline(model, data, device):
    print("\n" + "=" * 60)
    print("Stage 3 — Baseline Student (no distillation)")
    print("=" * 60)

    dataset      = TextDataset(data, block_size)
    split_idx    = int(0.9 * len(dataset))
    train_data   = torch.utils.data.Subset(dataset, range(0, split_idx))
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    optimizer    = torch.optim.AdamW(model.parameters(), lr=STUDENT_LR)

    model.train()
    step = 0

    while step < STUDENT_STEPS:
        for x, y in train_loader:
            if step >= STUDENT_STEPS:
                break

            x, y    = x.to(device), y.to(device)
            logits  = model(x)
            B, T, C = logits.shape
            loss    = F.cross_entropy(logits.view(B * T, C), y.view(B * T))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 500 == 0:
                print(f"  Baseline step {step}, Loss: {loss.item():.4f}")

            step += 1

    print("Baseline training complete")


# ─── Evaluation ───────────────────────────────────────────────────────────────

def evaluate(model, data, device, label="Model"):
    model.eval()
    dataset    = TextDataset(data, block_size)
    split_idx  = int(0.9 * len(dataset))
    val_data   = torch.utils.data.Subset(dataset, range(split_idx, len(dataset)))
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

    total_loss    = 0
    total_batches = 0

    with torch.no_grad():
        for x, y in val_loader:
            x, y    = x.to(device), y.to(device)
            logits  = model(x)
            B, T, C = logits.shape
            loss    = F.cross_entropy(logits.view(B * T, C), y.view(B * T))
            total_loss    += loss.item()
            total_batches += 1

    avg_loss = total_loss / total_batches
    print(f"  {label}: val loss = {avg_loss:.4f}")
    return avg_loss


# ─── Generation ───────────────────────────────────────────────────────────────

def generate(model, prompt, max_new_tokens=150, temperature=0.8):
    model.eval()
    tokens = tokenizer.encode(prompt)
    x      = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            x_cond     = x[:, -block_size:]
            logits     = model(x_cond)
            logits     = logits[:, -1, :] / temperature
            probs      = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            x          = torch.cat([x, next_token], dim=1)

    return tokenizer.decode(x[0].tolist())


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(123)

    text = download_corpus()
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    print(f"Vocab size:   {tokenizer.n_vocab:,}")
    print(f"Total tokens: {len(data):,}")

    teacher = GPT(
        vocab_size  = tokenizer.n_vocab,
        block_size  = block_size,
        embed_dim   = TEACHER_EMBED_DIM,
        num_heads   = TEACHER_NUM_HEADS,
        hidden_dim  = TEACHER_HIDDEN_DIM,
        num_layers  = TEACHER_NUM_LAYERS
    ).to(device)

    teacher_params = sum(p.numel() for p in teacher.parameters())
    print(f"\nTeacher parameters: {teacher_params:,}")

    student = GPT(
        vocab_size  = tokenizer.n_vocab,
        block_size  = block_size,
        embed_dim   = STUDENT_EMBED_DIM,
        num_heads   = STUDENT_NUM_HEADS,
        hidden_dim  = STUDENT_HIDDEN_DIM,
        num_layers  = STUDENT_NUM_LAYERS
    ).to(device)

    student_params = sum(p.numel() for p in student.parameters())
    print(f"Student parameters: {student_params:,}")
    print(f"Compression ratio:  {teacher_params / student_params:.1f}x\n")

    projection = FeatureProjection(STUDENT_EMBED_DIM, TEACHER_EMBED_DIM).to(device)

    baseline = GPT(
        vocab_size  = tokenizer.n_vocab,
        block_size  = block_size,
        embed_dim   = STUDENT_EMBED_DIM,
        num_heads   = STUDENT_NUM_HEADS,
        hidden_dim  = STUDENT_HIDDEN_DIM,
        num_layers  = STUDENT_NUM_LAYERS
    ).to(device)

    train_teacher(teacher, data, device)
    train_student(student, teacher, projection, data, device)
    train_baseline(baseline, data, device)

    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    teacher_loss  = evaluate(teacher,  data, device, "Teacher")
    student_loss  = evaluate(student,  data, device, "Distilled student (features)")
    baseline_loss = evaluate(baseline, data, device, "Baseline student")

    print(f"\n  Gap (distilled vs baseline): {baseline_loss - student_loss:.4f}")
    print(f"  Gap (teacher vs distilled):  {student_loss - teacher_loss:.4f}")

    print("\n" + "=" * 60)
    print("Generation Comparison")
    print("=" * 60)

    prompt = "Holmes examined the evidence carefully and concluded that"
    print(f"Prompt: {prompt}\n")

    print("Teacher:")
    print(generate(teacher, prompt))

    print("\nDistilled Student (features):")
    print(generate(student, prompt))

    print("\nBaseline Student:")
    print(generate(baseline, prompt))