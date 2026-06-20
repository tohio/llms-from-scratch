import os
import re
import json
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tiktoken
import urllib.request
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import pipeline


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
CORPUS_URLS = {
    "sherlock": "https://www.gutenberg.org/files/1661/1661-0.txt",
    "darwin":   "https://www.gutenberg.org/files/1228/1228-0.txt"
}
CORPUS_PATHS = {
    "sherlock": "../data/sherlock_corpus.txt",
    "darwin":   "../data/darwin_corpus.txt"
}
COMBINED_CORPUS_PATH = "../data/reasoning_corpus.txt"

# CommonsenseQA traces — generated once and cached to disk
# generation uses flan-t5-base — small enough to run locally
# production systems use larger teacher models (GPT-4, Llama-3-70B etc.)
TRACES_PATH  = "../data/commonsenseqa_traces.jsonl"
MODEL_SAVE   = "../data/reasoning_model_model.pt"

NUM_SAMPLES  = 1000   # how many CommonsenseQA examples to use
                       # raise for better coverage, lower for faster iteration
GRPO_GROUPS  = 4      # number of responses sampled per question in GRPO
                       # more groups = lower variance but slower
                       # DeepSeek R1 used 8 — 4 is a reasonable tutorial default
BATCH_SIZE   = 8


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
    # Download Sherlock + Darwin — same combined corpus as reasoning_prompt.py
    # reuses cached files if already downloaded
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
    return full_text


# ─── Trace Generation — SFT Data Preparation ─────────────────────────────────

def format_choices(choices):
    # Format multiple choice options as a readable string
    labels = choices["label"]
    texts  = choices["text"]
    return " ".join(f"{l}) {t}" for l, t in zip(labels, texts))


def generate_traces(n_samples):
    # Generate reasoning traces for CommonsenseQA using flan-t5-base
    # flan-t5-base is instruction-tuned — it can produce step-by-step reasoning
    # without needing few-shot prompting
    # traces are saved to disk after generation so this only runs once
    # production systems use larger teacher models for higher quality traces

    if os.path.exists(TRACES_PATH):
        print(f"Traces already exist at {TRACES_PATH} — loading from disk")
        traces = []
        with open(TRACES_PATH, "r", encoding="utf8") as f:
            for line in f:
                traces.append(json.loads(line.strip()))
        print(f"Loaded {len(traces)} traces")
        return traces

    print(f"\nGenerating reasoning traces for {n_samples} CommonsenseQA examples...")
    print("Using flan-t5-base — this runs once and caches to disk\n")

    # Load flan-t5-base on CPU — small enough, avoids device conflicts
    # alternative: use a larger model (flan-t5-xl, Mistral-7B) for better traces
    generator = pipeline(
        "text2text-generation",
        model  = "google/flan-t5-base",
        device = -1    # CPU — avoids device conflicts with the main model
    )

    dataset = load_dataset("tau/commonsense_qa", split=f"train[:{n_samples}]")

    traces = []
    for i, example in enumerate(dataset):
        question       = example["question"]
        choices_str    = format_choices(example["choices"])
        answer_key     = example["answerKey"]

        # Ask flan-t5 to reason through the question step by step
        # the prompt format follows the original CoT paper (Wei et al. 2022)
        prompt = (
            f"Question: {question}\n"
            f"Choices: {choices_str}\n"
            f"Let's think step by step."
        )

        try:
            result = generator(prompt, max_new_tokens=128, do_sample=False)[0]
            trace  = result["generated_text"].strip()
        except Exception:
            # if generation fails fall back to a minimal trace
            trace = f"The answer is {answer_key}."

        # Full SFT format: question + choices + trace + final answer
        # <think> tags mirror the R1 format — easy to extract answer at inference
        sft_text = (
            f"Question: {question}\n"
            f"Choices: {choices_str}\n"
            f"<think>\n{trace}\n</think>\n"
            f"Answer: {answer_key}"
        )

        traces.append({
            "question":   question,
            "choices":    choices_str,
            "answer":     answer_key,
            "trace":      trace,
            "sft_text":   sft_text
        })

        if (i + 1) % 100 == 0:
            print(f"  Generated {i + 1}/{n_samples} traces")

    # Save to disk — subsequent runs skip generation entirely
    with open(TRACES_PATH, "w", encoding="utf8") as f:
        for t in traces:
            f.write(json.dumps(t) + "\n")

    print(f"\nTraces saved to {TRACES_PATH}")
    return traces


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
    def __init__(self, traces, tokenizer, block_size):
        self.examples   = []
        self.block_size = block_size

        for trace in traces:
            tokens = tokenizer.encode(trace["sft_text"])

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


# ─── Stage 2 — SFT on Reasoning Traces ───────────────────────────────────────

def sft_train(model, traces, device):
    # Fine tune on CommonsenseQA reasoning traces
    # teaches the model the <think>...</think> + Answer: X format
    # the model learns to produce reasoning before committing to an answer
    print("\n" + "=" * 60)
    print("Stage 2 — SFT on CommonsenseQA reasoning traces")
    print("=" * 60)

    dataset   = SFTDataset(traces, tokenizer, block_size)
    loader    = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Freeze embeddings — preserve pretraining representations
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


# ─── Stage 3 — GRPO ───────────────────────────────────────────────────────────

def extract_answer(text):
    # Extract the answer letter (A-E) from the model's generated text
    # looks for "Answer: X" pattern produced by the SFT format
    # falls back to searching for any standalone A-E letter
    match = re.search(r"Answer:\s*([A-E])", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # fallback — look for any lone answer letter near the end
    match = re.search(r"\b([A-E])\b", text[-50:], re.IGNORECASE)
    if match:
        return match.group(1).upper()

    return None


def generate_response(model, prompt, max_new_tokens=150, temperature=0.8):
    # Sample a response from the model — used during GRPO to generate
    # the group of G responses per question
    model.eval()
    tokens = tokenizer.encode(prompt)

    # Truncate prompt if needed to leave room for the response
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


def compute_log_probs_for_response(model, prompt_tokens, response_tokens):
    # Compute log probability of a response sequence given the prompt
    # used to compute the GRPO policy gradient
    all_tokens = torch.tensor(
        prompt_tokens + response_tokens,
        dtype=torch.long,
        device=device
    ).unsqueeze(0)

    # Truncate to block_size
    if all_tokens.shape[1] > block_size:
        all_tokens = all_tokens[:, -block_size:]

    model.train()
    logits    = model(all_tokens)
    log_probs = F.log_softmax(logits, dim=-1)

    # Only gather log probs over the response portion
    prompt_len   = min(len(prompt_tokens), block_size - len(response_tokens))
    response_len = min(len(response_tokens), block_size - prompt_len)

    if response_len <= 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    # Gather log probs for each response token
    target_tokens = all_tokens[:, prompt_len:prompt_len + response_len]
    response_log_probs = log_probs[:, prompt_len - 1:prompt_len + response_len - 1, :]
    token_log_probs = response_log_probs.gather(
        2, target_tokens.unsqueeze(-1)
    ).squeeze(-1)

    return token_log_probs.sum()


def grpo_train(model, traces, device):
    # GRPO — Group Relative Policy Optimization
    # Used by DeepSeek R1 to train reasoning without a separate reward model
    #
    # For each question:
    #   1. Sample G responses from the current policy
    #   2. Score each response: +1 if correct answer, 0 if wrong
    #   3. Normalize scores within the group (subtract mean, divide by std)
    #      This is the "relative" in GRPO — rewards are relative to the group,
    #      not absolute. This stabilizes training and removes the need for a
    #      separate value/critic model (unlike PPO which needs one).
    #   4. Update the policy to increase probability of high-reward responses
    #
    # Alternatives:
    #   PPO    — needs a separate critic model, more stable but more complex
    #   REINFORCE — simpler than both but high variance, no relative scoring
    #   DPO    — needs pre-existing chosen/rejected pairs, no online sampling
    #
    # GRPO sits in the sweet spot: online sampling (like PPO/REINFORCE) but
    # with relative reward normalization that removes the need for a critic.

    print("\n" + "=" * 60)
    print("Stage 3 — GRPO on CommonsenseQA")
    print("=" * 60)
    print(f"Group size (G): {GRPO_GROUPS}")
    print(f"Reward: +1 correct answer, 0 wrong\n")

    optimizer = optim.AdamW(model.parameters(), lr=5e-6)  # very low lr for RL

    step          = 0
    total_correct = 0
    total_seen    = 0

    while step < grpo_max_steps:
        # Sample a random question from the traces
        trace = random.choice(traces)

        prompt = (
            f"Question: {trace['question']}\n"
            f"Choices: {trace['choices']}\n"
            f"<think>\n"
        )
        correct_answer = trace["answer"]
        prompt_tokens  = tokenizer.encode(prompt)

        # ── Step 1: Sample G responses ────────────────────────────────────────
        responses = []
        rewards   = []

        for _ in range(GRPO_GROUPS):
            response_text   = generate_response(model, prompt)
            # Extract only the generated portion
            generated       = response_text[len(prompt):]
            predicted       = extract_answer(generated)
            reward          = 1.0 if predicted == correct_answer else 0.0

            responses.append(generated)
            rewards.append(reward)

        # ── Step 2: Normalize rewards within the group ────────────────────────
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)

        if rewards_tensor.std() > 1e-8:
            # Subtract mean and divide by std — relative scoring
            # if all responses are correct or all wrong, std ≈ 0
            # skip the update in that case — no signal to learn from
            normalized_rewards = (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std() + 1e-8)
        else:
            # All responses gave the same reward — no relative signal
            # skip this question
            step += 1
            continue

        # ── Step 3: Policy gradient update ───────────────────────────────────
        # Maximize expected reward — increase log prob of high-reward responses
        # decrease log prob of low-reward responses
        loss = torch.tensor(0.0, device=device, requires_grad=True)

        for response, norm_reward in zip(responses, normalized_rewards.tolist()):
            response_tokens = tokenizer.encode(response)
            if not response_tokens:
                continue

            log_prob  = compute_log_probs_for_response(model, prompt_tokens, response_tokens)
            # GRPO loss: negative because we want to maximize reward
            # multiply log prob by normalized reward
            # high positive reward → increase log prob of this response
            # high negative reward → decrease log prob of this response
            loss = loss + (-log_prob * norm_reward)

        loss = loss / GRPO_GROUPS

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Track accuracy
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
    # Lower temperature than pretraining — reasoning benefits from
    # more deterministic output
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

    # Pretraining corpus
    text = download_corpus()
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    print(f"Pretraining tokens: {len(data):,}")

    # CommonsenseQA traces — generated once then cached
    traces = generate_traces(NUM_SAMPLES)
    print(f"SFT/GRPO examples: {len(traces)}")

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

    # Test after pretraining
    print("After pretraining:")
    print(generate(model, "Holmes looked at the evidence and concluded"))
    print()

    # ── Stage 2 — SFT on reasoning traces ────────────────────────────────────
    sft_train(model, traces, device)

    # Test after SFT — should start producing <think> format
    test_q  = traces[0]
    prompt  = (
        f"Question: {test_q['question']}\n"
        f"Choices: {test_q['choices']}\n"
        f"<think>\n"
    )
    print("After SFT:")
    print(generate(model, prompt))
    print()

    # ── Stage 3 — GRPO ────────────────────────────────────────────────────────
    grpo_train(model, traces, device)

    # Test after GRPO — should show improved answer accuracy
    print("After GRPO:")
    for trace in traces[:3]:
        prompt = (
            f"Question: {trace['question']}\n"
            f"Choices: {trace['choices']}\n"
            f"<think>\n"
        )
        response  = generate(model, prompt)
        predicted = extract_answer(response)
        correct   = trace["answer"]
        status    = "✓" if predicted == correct else "✗"
        print(f"  {status} Predicted: {predicted} | Correct: {correct}")
        print(f"  Q: {trace['question'][:60]}...")
        print()

    # ── Save ──────────────────────────────────────────────────────────────────
    torch.save(model.state_dict(), MODEL_SAVE)
    print(f"Model saved to {MODEL_SAVE}")