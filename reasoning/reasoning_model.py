import os
import re
import random
import urllib.request
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tiktoken
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset


# ─── Hardware Config ──────────────────────────────────────────────────────────

device = torch.device(
    "mps"  if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available()         else
    "cpu"
)

print(f"Using device: {device}")

# ── Laptop / GPU < 40GB VRAM ──
embed_dim          = 256
num_heads          = 8
hidden_dim         = 1024
num_layers         = 8
block_size         = 256    # longer — reasoning traces are verbose
pretrain_max_steps = 10000
sft_max_steps      = 3000
grpo_max_steps     = 1000

# ── Cloud GPU ≥ 40GB VRAM (A100, H100) ──
# embed_dim          = 512
# num_heads          = 16
# hidden_dim         = 2048
# num_layers         = 12
# block_size         = 512
# pretrain_max_steps = 50000
# sft_max_steps      = 10000
# grpo_max_steps     = 5000
# USE_COMPILE        = True
# USE_AMP            = True

# ── CPU / Small GPU (Tesla V100) ──
# embed_dim          = 128
# num_heads          = 4
# hidden_dim         = 512
# num_layers         = 4
# block_size         = 128
# pretrain_max_steps = 5000
# sft_max_steps      = 1000
# grpo_max_steps     = 500


# ─── Config ───────────────────────────────────────────────────────────────────

# Pretraining corpora — Sherlock + Darwin, same as reasoning_prompt.py
# general language understanding before reasoning training
CORPUS_URLS = {
    "sherlock": "https://www.gutenberg.org/files/1661/1661-0.txt",
    "darwin":   "https://www.gutenberg.org/files/1228/1228-0.txt"
}
CORPUS_PATHS = {
    "sherlock": "../data/sherlock_corpus.txt",
    "darwin":   "../data/darwin_corpus.txt"
}
COMBINED_CORPUS_PATH = "../data/reasoning_corpus.txt"
MODEL_SAVE           = "../data/reasoning_model.pt"

# Nemotron dataset config
# chat — teaches <think> format and controllable reasoning on/off
# math — provides verifiable answers for GRPO reward signal
# both filtered to Nano subset — pre-vetted high quality samples
# used_in_training == "Nano" is the smallest vetted tier in the Nemotron family
NEMOTRON_DATASET  = "nvidia/Llama-Nemotron-Post-Training-Dataset"
SFT_SPLIT         = "chat"     # 39k samples — teaches reasoning format
GRPO_SPLIT        = "math"     # 22M samples — verifiable math answers for RL
NUM_SFT_SAMPLES   = 500        # stream and take this many chat examples
NUM_GRPO_SAMPLES  = 200        # stream and take this many math examples

BATCH_SIZE        = 8
GRPO_GROUPS       = 4          # responses sampled per question in GRPO
                                # more groups = lower variance, slower per step
                                # DeepSeek R1 used 8 — 4 is a good tutorial default

# Controllable reasoning system prompts — from the Nemotron paper
# the model learns to toggle reasoning on/off based on the system prompt
# this is the key innovation of Llama Nemotron vs standard reasoning models
THINKING_ON  = "detailed thinking on"
THINKING_OFF = "detailed thinking off"


# ─── Tokenizer ────────────────────────────────────────────────────────────────

tokenizer = tiktoken.encoding_for_model("gpt-4o")


# ─── Corpus — Pretrain ────────────────────────────────────────────────────────

def strip_gutenberg(text):
    start = text.find("*** START OF")
    end   = text.find("*** END OF")
    if start != -1 and end != -1:
        text = text[start:end]
    return text.strip()


def download_corpus():
    # Download Sherlock + Darwin — reuses cached files if already present
    # same combined corpus as reasoning_prompt.py
    combined = []

    for name, url in CORPUS_URLS.items():
        path = CORPUS_PATHS[name]

        if not os.path.exists(path):
            print(f"Downloading {name} corpus...")
            urllib.request.urlretrieve(url, path)
        else:
            print(f"{name} corpus already exists at {path}")

        with open(path, "r", encoding="utf8") as f:
            text = f.read()

        text = strip_gutenberg(text)
        combined.append(text)
        print(f"{name}: {len(text):,} characters")

    if os.path.exists(COMBINED_CORPUS_PATH):
        print(f"Combined corpus already exists at {COMBINED_CORPUS_PATH}")
        with open(COMBINED_CORPUS_PATH, "r", encoding="utf8") as f:
            return f.read()

    full_text = "\n\n---\n\n".join(combined)

    with open(COMBINED_CORPUS_PATH, "w", encoding="utf8") as f:
        f.write(full_text)

    print(f"Combined corpus: {len(full_text):,} characters")
    return full_text


# ─── Nemotron Data Loading ─────────────────────────────────────────────────────

def load_sft_data(n_samples):
    # Load chat split from Nemotron dataset
    # filter to Nano-vetted samples only — highest quality tier
    # balance reasoning on/off — teaches controllable reasoning
    # skip samples with empty <think></think> — these are refusal samples
    print(f"\nLoading Nemotron SFT data ({SFT_SPLIT} split, Nano subset)...")

    dataset = load_dataset(
        NEMOTRON_DATASET,
        "SFT",
        split     = SFT_SPLIT,
        streaming = True
    )

    on_samples  = []
    off_samples = []

    for example in dataset:
        if len(on_samples) + len(off_samples) >= n_samples * 2:
            break

        # Filter to Nano-vetted samples only
        used_in = example.get("used_in_training", "")
        if "Nano" not in used_in:
            continue

        reasoning     = example.get("reasoning", "")
        output        = example.get("output",    "")
        input_msgs    = example.get("input",     [])
        system_prompt = example.get("system_prompt", "")

        if not input_msgs or not output:
            continue

        # Skip empty think tags — these are typically refusal samples
        if reasoning == "on" and "<think></think>" in output:
            continue

        # Extract user message from the input list
        user_msg = ""
        for msg in input_msgs:
            if isinstance(msg, dict) and msg.get("role") == "user":
                user_msg = msg.get("content", "")
                break

        if not user_msg:
            continue

        sample = {
            "system_prompt": system_prompt,
            "user":          user_msg,
            "output":        output,
            "reasoning":     reasoning
        }

        if reasoning == "on" and len(on_samples) < n_samples // 2:
            on_samples.append(sample)
        elif reasoning == "off" and len(off_samples) < n_samples // 2:
            off_samples.append(sample)

    # Interleave on/off samples — curriculum: easier off samples mixed with harder on samples
    samples = []
    for on, off in zip(on_samples, off_samples):
        samples.append(off)   # reasoning off first — simpler, no trace needed
        samples.append(on)    # reasoning on second — requires <think> trace

    print(f"Loaded {len(samples)} SFT samples ({len(on_samples)} reasoning on, {len(off_samples)} reasoning off)")
    return samples


def load_grpo_data(n_samples):
    # Load math split for GRPO — math has verifiable \boxed{} answers
    # GRPO needs a ground truth reward signal — format reward alone is not enough
    # math answers are objectively right or wrong — perfect for binary reward
    print(f"\nLoading Nemotron GRPO data ({GRPO_SPLIT} split, Nano subset)...")

    dataset = load_dataset(
        NEMOTRON_DATASET,
        "SFT",
        split     = GRPO_SPLIT,
        streaming = True
    )

    samples = []

    for example in dataset:
        if len(samples) >= n_samples:
            break

        used_in   = example.get("used_in_training", "")
        reasoning = example.get("reasoning", "")
        output    = example.get("output",    "")
        input_msgs = example.get("input",    [])

        # GRPO only on reasoning-on math samples — we need <think> traces + \boxed{} answers
        if "Nano" not in used_in:
            continue
        if reasoning != "on":
            continue
        if not output or "<think></think>" in output:
            continue
        if not extract_boxed_answer(output):
            continue    # skip if no \boxed{} answer — can't verify

        user_msg = ""
        for msg in input_msgs:
            if isinstance(msg, dict) and msg.get("role") == "user":
                user_msg = msg.get("content", "")
                break

        if not user_msg:
            continue

        samples.append({
            "user":   user_msg,
            "output": output,
            "answer": extract_boxed_answer(output)
        })

    print(f"Loaded {len(samples)} GRPO math samples")
    return samples


# ─── Datasets ─────────────────────────────────────────────────────────────────

class PretrainDataset(Dataset):
    def __init__(self, data, block_size):
        self.data       = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.block_size]
        y = self.data[idx + 1:idx + self.block_size + 1]
        return x, y


class SFTDataset(Dataset):
    def __init__(self, samples, tokenizer, block_size):
        # Format each sample as:
        # System: {system_prompt}
        # User: {user}
        # Assistant: {output}
        # The system prompt carries the reasoning toggle signal
        self.examples   = []
        self.block_size = block_size

        for sample in samples:
            text = (
                f"System: {sample['system_prompt']}\n"
                f"User: {sample['user']}\n"
                f"Assistant: {sample['output']}"
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


# ─── Stage 1 — Pretraining ────────────────────────────────────────────────────

def pretrain(model, data, device):
    # Train on Sherlock + Darwin — general language and reasoning patterns
    # before SFT the model has no instruction following capability
    # pretraining gives it the language foundation to build on
    print("\n" + "=" * 60)
    print("Stage 1 — Pretraining on Sherlock + Darwin")
    print("=" * 60)

    dataset      = PretrainDataset(data, block_size)
    split_idx    = int(0.9 * len(dataset))
    train_data   = torch.utils.data.Subset(dataset, range(0, split_idx))
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    optimizer    = optim.AdamW(model.parameters(), lr=3e-4)

    model.train()
    step = 0

    while step < pretrain_max_steps:
        for x, y in train_loader:
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
                print(f"  Step {step}, Loss: {loss.item():.4f}")

            step += 1

    print("Pretraining complete\n")


# ─── Stage 2 — SFT on Nemotron Chat ──────────────────────────────────────────

def sft_train(model, sft_samples, device):
    # Fine tune on Nemotron chat data — teaches two things:
    # 1. The <think>...</think> format for reasoning traces
    # 2. Controllable reasoning — respond to "detailed thinking on/off"
    #    in the system prompt to toggle reasoning mode
    #
    # Samples are interleaved on/off so the model sees both modes equally
    # curriculum: easier off samples come before harder on samples in each pair
    print("\n" + "=" * 60)
    print("Stage 2 — SFT on Nemotron chat (controllable reasoning)")
    print("=" * 60)

    dataset   = SFTDataset(sft_samples, tokenizer, block_size)
    loader    = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)  # keep curriculum order

    # Freeze embeddings — preserve pretraining token representations
    for param in model.token_embedding.parameters():
        param.requires_grad = False
    for param in model.position_embedding.parameters():
        param.requires_grad = False

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4
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
            loss    = F.cross_entropy(logits.view(B * T, C), y.view(B * T))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 200 == 0:
                print(f"  SFT step {step}, Loss: {loss.item():.4f}")

            step += 1

    # Unfreeze embeddings
    for param in model.token_embedding.parameters():
        param.requires_grad = True
    for param in model.position_embedding.parameters():
        param.requires_grad = True

    print("SFT complete\n")


# ─── Stage 3 — GRPO on Nemotron Math ─────────────────────────────────────────

def extract_boxed_answer(text):
    # Extract the answer from LaTeX \boxed{} notation
    # standard format for math answers in the Nemotron dataset
    # handles nested braces e.g. \boxed{\frac{1}{2}}
    match = re.search(r"\\boxed\{([^}]*(?:\{[^}]*\}[^}]*)*)\}", text)
    return match.group(1).strip() if match else None


def generate_response(model, prompt, max_new_tokens=150, temperature=0.8):
    # Sample a response for GRPO — temperature > 0 ensures diversity
    # across the G responses sampled per question
    model.eval()
    tokens = tokenizer.encode(prompt)

    if len(tokens) > block_size - max_new_tokens:
        tokens = tokens[-(block_size - max_new_tokens):]

    x = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            x_cond     = x[:, -block_size:]
            logits     = model(x_cond)
            logits     = logits[:, -1, :] / temperature
            probs      = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            x          = torch.cat([x, next_token], dim=1)

    return tokenizer.decode(x[0].tolist())


def compute_response_log_prob(model, prompt_tokens, response_tokens):
    # Compute the total log probability of the response tokens given the prompt
    # used to compute the GRPO policy gradient update
    all_tokens = torch.tensor(
        prompt_tokens + response_tokens,
        dtype=torch.long,
        device=device
    ).unsqueeze(0)

    if all_tokens.shape[1] > block_size:
        all_tokens = all_tokens[:, -block_size:]

    model.train()
    logits    = model(all_tokens)
    log_probs = F.log_softmax(logits, dim=-1)

    prompt_len   = min(len(prompt_tokens), block_size - len(response_tokens))
    response_len = min(len(response_tokens), block_size - prompt_len)

    if response_len <= 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    target_tokens      = all_tokens[:, prompt_len:prompt_len + response_len]
    response_log_probs = log_probs[:, prompt_len - 1:prompt_len + response_len - 1, :]
    token_log_probs    = response_log_probs.gather(2, target_tokens.unsqueeze(-1)).squeeze(-1)

    return token_log_probs.sum()


def grpo_train(model, grpo_samples, device):
    # GRPO — Group Relative Policy Optimization
    # Used by DeepSeek R1 to train reasoning without a separate reward model
    #
    # For each math question:
    #   1. Sample G=4 responses from the current policy
    #   2. Score each: +1 if \boxed{} answer matches ground truth, 0 if wrong
    #   3. Normalize rewards within the group (subtract mean, divide by std)
    #      relative scoring removes the need for a value/critic model (unlike PPO)
    #      if all responses are correct or all wrong — no update (no signal)
    #   4. Policy gradient: increase log prob of high-reward responses
    #
    # Why math for GRPO and not chat:
    #   Chat responses have no ground truth — you can't verify if they're correct
    #   Math answers are objectively right or wrong — perfect binary reward signal
    #   Format reward alone (rewarding <think> tags) doesn't verify reasoning quality
    #
    # Alternatives:
    #   PPO       — needs a separate critic/value model, more stable but more complex
    #   REINFORCE — simpler but high variance, no relative reward normalization
    #   DPO       — needs pre-existing chosen/rejected pairs, no online sampling

    print("\n" + "=" * 60)
    print("Stage 3 — GRPO on Nemotron math")
    print("=" * 60)
    print(f"Group size G: {GRPO_GROUPS}")
    print(f"Reward: +1 correct \\boxed{{}} answer, 0 wrong\n")

    optimizer     = optim.AdamW(model.parameters(), lr=5e-6)
    step          = 0
    total_correct = 0
    total_seen    = 0

    while step < grpo_max_steps:
        sample         = random.choice(grpo_samples)
        correct_answer = sample["answer"]

        # Build prompt with "detailed thinking on" system prompt
        # GRPO always uses reasoning on — we want the model to think before answering
        prompt = (
            f"System: {THINKING_ON}\n"
            f"User: {sample['user']}\n"
            f"Assistant: <think>\n"
        )
        prompt_tokens = tokenizer.encode(prompt)

        # ── Step 1: Sample G responses ────────────────────────────────────────
        responses = []
        rewards   = []

        for _ in range(GRPO_GROUPS):
            response_text   = generate_response(model, prompt)
            generated       = response_text[len(prompt):]
            predicted       = extract_boxed_answer(generated)
            reward          = 1.0 if (predicted is not None and predicted == correct_answer) else 0.0

            responses.append(generated)
            rewards.append(reward)

        # ── Step 2: Normalize rewards within the group ────────────────────────
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)

        if rewards_tensor.std() < 1e-8:
            # All responses gave the same reward — no relative signal to learn from
            # skip this question rather than making a noisy update
            step += 1
            continue

        normalized_rewards = (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std() + 1e-8)

        # ── Step 3: Policy gradient update ───────────────────────────────────
        loss = torch.tensor(0.0, device=device, requires_grad=True)

        for response, norm_reward in zip(responses, normalized_rewards.tolist()):
            response_tokens = tokenizer.encode(response)
            if not response_tokens:
                continue

            log_prob = compute_response_log_prob(model, prompt_tokens, response_tokens)
            # Negative because optimizer minimizes — we want to maximize reward
            # high positive norm_reward → increase log prob of this response
            # high negative norm_reward → decrease log prob of this response
            loss = loss + (-log_prob * norm_reward)

        loss = loss / GRPO_GROUPS

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_correct += sum(rewards)
        total_seen    += GRPO_GROUPS

        if step % 50 == 0:
            accuracy = total_correct / max(total_seen, 1)
            print(f"  GRPO step {step}, Loss: {loss.item():.4f}, "
                  f"Accuracy: {accuracy:.2%} ({int(total_correct)}/{total_seen})")
            total_correct = 0
            total_seen    = 0

        step += 1

    print("GRPO complete\n")


# ─── Generation ───────────────────────────────────────────────────────────────

def generate(model, prompt, max_new_tokens=200, temperature=0.7):
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

    # ── Stage 0 — Data preparation ────────────────────────────────────────────
    print("=" * 60)
    print("Stage 0 — Data Preparation")
    print("=" * 60)

    # Pretraining corpus — Sherlock + Darwin
    text = download_corpus()
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    print(f"Pretraining tokens: {len(data):,}")

    # Nemotron SFT data — chat, reasoning on/off balanced
    sft_samples = load_sft_data(NUM_SFT_SAMPLES)

    # Nemotron GRPO data — math, reasoning on only
    grpo_samples = load_grpo_data(NUM_GRPO_SAMPLES)

    # ── Build model ───────────────────────────────────────────────────────────
    model = GPT(
        vocab_size  = tokenizer.n_vocab,
        block_size  = block_size,
        embed_dim   = embed_dim,
        num_heads   = num_heads,
        hidden_dim  = hidden_dim,
        num_layers  = num_layers
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {total_params:,}")

    # ── Stage 1 — Pretrain ────────────────────────────────────────────────────
    pretrain(model, data, device)

    # ── Stage 2 — SFT on Nemotron chat ───────────────────────────────────────
    sft_train(model, sft_samples, device)

    # Test controllable reasoning after SFT
    print("After SFT — reasoning ON:")
    prompt_on = (
        f"System: {THINKING_ON}\n"
        f"User: What is the capital of France?\n"
        f"Assistant:"
    )
    print(generate(model, prompt_on))

    print("\nAfter SFT — reasoning OFF:")
    prompt_off = (
        f"System: {THINKING_OFF}\n"
        f"User: What is the capital of France?\n"
        f"Assistant:"
    )
    print(generate(model, prompt_off))
    print()

    # ── Stage 3 — GRPO on Nemotron math ──────────────────────────────────────
    grpo_train(model, grpo_samples, device)

    # Test after GRPO — should show improved math reasoning
    print("After GRPO — math reasoning:")
    for sample in grpo_samples[:3]:
        prompt = (
            f"System: {THINKING_ON}\n"
            f"User: {sample['user'][:200]}\n"
            f"Assistant: <think>\n"
        )
        response  = generate(model, prompt)
        predicted = extract_boxed_answer(response)
        correct   = sample["answer"]
        status    = "✓" if predicted == correct else "✗"
        print(f"  {status} Predicted: {predicted} | Correct: {correct}")
        print(f"  Q: {sample['user'][:80]}...")
        print()

    # ── Save ──────────────────────────────────────────────────────────────────
    torch.save(model.state_dict(), MODEL_SAVE)
    print(f"Model saved to {MODEL_SAVE}")