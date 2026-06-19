import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken
import urllib.request
import os
from torch.utils.data import Dataset, DataLoader


# ─── Hardware Config ──────────────────────────────────────────────────────────

# Device detection — automatically picks the best available
device = (
    "mps"  if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available()         else
    "cpu"
)

print(f"Using device: {device}")

# ── Laptop / M4 Max ──
# Teacher — larger model, trained first, then frozen
TEACHER_EMBED_DIM  = 256
TEACHER_NUM_HEADS  = 8
TEACHER_HIDDEN_DIM = 1024
TEACHER_NUM_LAYERS = 8

# Student — smaller model, learns from teacher
STUDENT_EMBED_DIM  = 128
STUDENT_NUM_HEADS  = 4
STUDENT_HIDDEN_DIM = 512
STUDENT_NUM_LAYERS = 4

block_size = 128

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
# USE_COMPILE        = True    # torch.compile — significant speedup on CUDA
# USE_AMP            = True    # automatic mixed precision — CUDA only, not MPS

# ── CPU / Small GPU ──
# TEACHER_EMBED_DIM  = 128
# TEACHER_NUM_HEADS  = 4
# TEACHER_HIDDEN_DIM = 512
# TEACHER_NUM_LAYERS = 4
# STUDENT_EMBED_DIM  = 64
# STUDENT_NUM_HEADS  = 2
# STUDENT_HIDDEN_DIM = 256
# STUDENT_NUM_LAYERS = 2
# block_size         = 64


# ─── Config ───────────────────────────────────────────────────────────────────

# Two complementary reasoning corpora from Project Gutenberg
# Sherlock Holmes — deductive reasoning (specific clues → single conclusion)
# Darwin — inductive reasoning (many observations → general theory)
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
STUDENT_SAVE         = "../data/distillation_logits_student.pt"

# Training hyperparameters
TEACHER_STEPS    = 10000
STUDENT_STEPS    = 5000
BATCH_SIZE       = 16
TEACHER_LR       = 3e-4
STUDENT_LR       = 1e-4

# Distillation hyperparameters
TEMPERATURE      = 4.0    # softens teacher distribution — higher = softer
                          # soft labels carry more information than hard labels
                          # temperature > 1 amplifies small probabilities
                          # temperature = 1 is standard softmax — no softening
ALPHA            = 0.5    # weight for hard label loss (cross entropy)
BETA             = 0.5    # weight for soft label loss (KL divergence)
                          # alpha + beta should sum to 1.0
                          # higher alpha — student learns more from ground truth
                          # higher beta  — student learns more from teacher


# ─── Tokenizer ────────────────────────────────────────────────────────────────

# Both teacher and student MUST use the same tokenizer
# logit distillation requires identical vocabulary sizes
# the teacher output is a distribution over vocab_size tokens
# the student must match that exact distribution — impossible with different vocabs
tokenizer = tiktoken.encoding_for_model("gpt-4o")


# ─── Corpus ───────────────────────────────────────────────────────────────────

def strip_gutenberg(text):
    # Strip Project Gutenberg header and footer
    start = text.find("*** START OF")
    end   = text.find("*** END OF")
    if start != -1 and end != -1:
        text = text[start:end]
    return text.strip()


def download_corpus():
    # Download both corpora — only downloads if not already present
    # reuses files downloaded by reasoning.py if they exist
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

    # Check if combined corpus already exists
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

    def forward(self, x):
        b, seq_len          = x.shape
        token_embeddings    = self.token_embedding(x)
        positions           = torch.arange(seq_len, device=x.device)
        position_embeddings = self.position_embedding(positions)
        x = token_embeddings + position_embeddings
        x = self.blocks(x)
        x = self.final_norm(x)
        return self.lm_head(x)


# ─── Train Teacher ────────────────────────────────────────────────────────────

def train_teacher(model, data, device):
    # Train the teacher model from scratch on the combined corpus
    # teacher is larger and trains longer — it becomes the knowledge source
    # once trained the teacher is frozen and never updated again
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


# ─── Logit Distillation Loss ──────────────────────────────────────────────────

def distillation_loss(student_logits, teacher_logits, targets, temperature, alpha, beta):
    # Logit distillation — the student learns from two signals simultaneously:
    #
    # 1. Hard label loss — standard cross entropy against ground truth tokens
    #    this is the same loss used in normal language model training
    #
    # 2. Soft label loss — KL divergence between student and teacher distributions
    #    the teacher's soft probabilities carry richer information than hard labels
    #    example: for the token "dog", the teacher might assign:
    #      "dog": 0.6, "cat": 0.2, "animal": 0.1, "pet": 0.05 ...
    #    this tells the student that "cat" and "animal" are related — hard labels cannot
    #
    # temperature scaling softens both distributions before computing KL divergence
    # higher temperature = flatter distribution = more information in the soft labels
    # at inference time temperature is set back to 1.0

    B, T, C = student_logits.shape

    # Hard label loss — standard next token prediction
    hard_loss = F.cross_entropy(
        student_logits.view(B * T, C),
        targets.view(B * T)
    )

    # Soft label loss — KL divergence between temperature-scaled distributions
    # temperature² scaling factor restores gradient magnitudes after softening
    # this is from the original Hinton et al. 2015 distillation paper
    student_soft = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_soft = F.softmax(teacher_logits   / temperature, dim=-1)

    # KL divergence — measures how much student distribution diverges from teacher
    # F.kl_div expects log probabilities for input and probabilities for target
    soft_loss = F.kl_div(
        student_soft.view(B * T, C),
        teacher_soft.view(B * T, C),
        reduction = "batchmean"
    ) * (temperature ** 2)

    # Combined loss — weighted sum of hard and soft label losses
    return alpha * hard_loss + beta * soft_loss, hard_loss, soft_loss


# ─── Train Student ────────────────────────────────────────────────────────────

def train_student(student, teacher, data, device):
    # Train the student model using logit distillation
    # teacher is frozen — no gradients flow through it
    # student learns from both ground truth tokens and teacher soft labels
    print("\n" + "=" * 60)
    print("Stage 2 — Logit Distillation")
    print("=" * 60)
    print(f"Temperature:  {TEMPERATURE}")
    print(f"Alpha (hard): {ALPHA}")
    print(f"Beta  (soft): {BETA}\n")

    dataset      = TextDataset(data, block_size)
    split_idx    = int(0.9 * len(dataset))
    train_data   = torch.utils.data.Subset(dataset, range(0, split_idx))
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    optimizer    = torch.optim.AdamW(student.parameters(), lr=STUDENT_LR)

    # Freeze teacher — it is purely a knowledge source
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False

    student.train()
    step = 0

    while step < STUDENT_STEPS:
        for x, y in train_loader:
            if step >= STUDENT_STEPS:
                break

            x, y = x.to(device), y.to(device)

            # Student forward pass — gradients flow
            student_logits = student(x)

            # Teacher forward pass — no gradients needed
            with torch.no_grad():
                teacher_logits = teacher(x)

            # Combined distillation loss
            loss, hard_loss, soft_loss = distillation_loss(
                student_logits, teacher_logits, y,
                TEMPERATURE, ALPHA, BETA
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 500 == 0:
                print(f"  Step {step} — Total: {loss.item():.4f} | Hard: {hard_loss.item():.4f} | Soft: {soft_loss.item():.4f}")

            step += 1

    torch.save(student.state_dict(), STUDENT_SAVE)
    print(f"\nStudent saved to {STUDENT_SAVE}")


# ─── Baseline Student (no distillation) ──────────────────────────────────────

def train_baseline(model, data, device):
    # Train a student-sized model from scratch without distillation
    # used as a baseline to measure the benefit of distillation
    # if distillation works the distilled student should outperform this baseline
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
    # Compute validation loss — lower is better
    # used to compare teacher, distilled student, and baseline student
    model.eval()
    dataset   = TextDataset(data, block_size)
    split_idx = int(0.9 * len(dataset))
    val_data  = torch.utils.data.Subset(dataset, range(split_idx, len(dataset)))
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

    total_loss = 0
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

    # ── Download and prepare corpus ───────────────────────────────────────────
    text = download_corpus()
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    print(f"Vocab size:   {tokenizer.n_vocab:,}")
    print(f"Total tokens: {len(data):,}")

    # ── Build teacher ─────────────────────────────────────────────────────────
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

    # ── Build distilled student ───────────────────────────────────────────────
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

    # ── Build baseline student (same size, no distillation) ───────────────────
    baseline = GPT(
        vocab_size  = tokenizer.n_vocab,
        block_size  = block_size,
        embed_dim   = STUDENT_EMBED_DIM,
        num_heads   = STUDENT_NUM_HEADS,
        hidden_dim  = STUDENT_HIDDEN_DIM,
        num_layers  = STUDENT_NUM_LAYERS
    ).to(device)

    # ── Stage 1 — Train teacher ───────────────────────────────────────────────
    train_teacher(teacher, data, device)

    # ── Stage 2 — Distil into student ─────────────────────────────────────────
    train_student(student, teacher, data, device)

    # ── Stage 3 — Train baseline (no distillation) ───────────────────────────
    train_baseline(baseline, data, device)

    # ── Comparison ────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    teacher_loss  = evaluate(teacher,  data, device, "Teacher")
    student_loss  = evaluate(student,  data, device, "Distilled student")
    baseline_loss = evaluate(baseline, data, device, "Baseline student")

    print(f"\n  Gap (distilled vs baseline): {baseline_loss - student_loss:.4f}")
    print(f"  Gap (teacher vs distilled):  {student_loss - teacher_loss:.4f}")

    # ── Generation comparison ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Generation Comparison")
    print("=" * 60)

    prompt = "Holmes examined the evidence carefully and concluded that"
    print(f"Prompt: {prompt}\n")

    print("Teacher:")
    print(generate(teacher, prompt))

    print("\nDistilled Student:")
    print(generate(student, prompt))

    print("\nBaseline Student:")
    print(generate(baseline, prompt))