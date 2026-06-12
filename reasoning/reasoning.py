import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken
import urllib.request
import os
import re
import copy
from collections import Counter
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
embed_dim  = 256
num_heads  = 8
hidden_dim = 1024
num_layers = 8
block_size = 128

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

# Corpus — swap URL and PATH to use a different book or dataset
# any plain text file from Project Gutenberg works here
CORPUS_URL  = "https://www.gutenberg.org/files/1661/1661-0.txt"
CORPUS_PATH = "../data/sherlock_corpus.txt"
MODEL_SAVE  = "../data/reasoning_model.pt"

batch_size    = 16
max_steps     = 10000
learning_rate = 3e-4


# ─── Tokenizer ────────────────────────────────────────────────────────────────

# GPT-4o BPE tokenizer — consistent with the rest of the repo
tokenizer = tiktoken.encoding_for_model("gpt-4o")


# ─── Corpus ───────────────────────────────────────────────────────────────────

def download_corpus():
    # Download corpus from Project Gutenberg
    # only downloads if not already present — subsequent runs skip this step
    # swap CORPUS_URL and CORPUS_PATH at the top to use a different book
    if not os.path.exists(CORPUS_PATH):
        print(f"Downloading corpus to {CORPUS_PATH}...")
        urllib.request.urlretrieve(CORPUS_URL, CORPUS_PATH)
        print("Download complete")
    else:
        print(f"Corpus already exists at {CORPUS_PATH}")

    with open(CORPUS_PATH, "r", encoding="utf8") as f:
        text = f.read()

    # Strip Project Gutenberg header and footer
    # the actual story starts and ends at the *** markers
    start = text.find("*** START OF")
    end   = text.find("*** END OF")
    if start != -1 and end != -1:
        text = text[start:end]

    print(f"Corpus size: {len(text):,} characters")
    return text


# ─── Dataset ──────────────────────────────────────────────────────────────────

class TextDataset(Dataset):
    def __init__(self, data, block_size):
        # data is a tensor of token IDs
        # block_size controls how many tokens the model sees at once
        self.data       = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        # x is the input sequence, y is shifted one position right
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


# ─── Training ─────────────────────────────────────────────────────────────────

def train(model, data, device):
    print("\n" + "=" * 60)
    print("Training")
    print("=" * 60)

    dataset      = TextDataset(data, block_size)
    split_idx    = int(0.9 * len(dataset))
    train_data   = torch.utils.data.Subset(dataset, range(0, split_idx))
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    optimizer    = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    model.train()
    step = 0

    while step < max_steps:
        for x, y in train_loader:
            if step >= max_steps:
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

    print("Training complete\n")
    torch.save(model.state_dict(), MODEL_SAVE)
    print(f"Model saved to {MODEL_SAVE}")


# ─── Base Generation ──────────────────────────────────────────────────────────

def generate(model, prompt, max_new_tokens=200, temperature=0.8):
    # Base generation used by all reasoning techniques
    # temperature=0.8 — slightly creative but not purely random
    # swap temperature per technique as needed
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


# ─── Technique 1 — Chain of Thought ──────────────────────────────────────────

def chain_of_thought(model):
    # Chain of Thought prompting — ask the model to reason step by step
    # before giving a final answer
    # the prompt format guides the model to show its reasoning process
    # CoT works best when the base model has seen reasoning patterns in training
    # Sherlock Holmes is ideal — he always observes → deduces → concludes
    # swap the prompt to test CoT on different reasoning domains
    print("=" * 60)
    print("Technique 1 — Chain of Thought (CoT)")
    print("=" * 60)
    print("CoT prompts the model to show its reasoning step by step")
    print("before arriving at a conclusion.\n")

    prompt = (
        "Holmes looked carefully at the visitor and said:\n"
        "Let me think through this step by step.\n"
        "First, I observe that"
    )

    print(f"Prompt:\n{prompt}\n")
    output = generate(model, prompt, max_new_tokens=300)
    print(f"Output:\n{output}\n")


# ─── Technique 2 — Self Consistency ──────────────────────────────────────────

def self_consistency(model, n_samples=5):
    # Self consistency — sample multiple reasoning paths from the same prompt
    # then take a majority vote on the conclusion
    # reduces the impact of any single bad generation
    # works well when the model can reach the same conclusion via different paths
    # the key insight: wrong reasoning paths tend to disagree with each other
    # while correct paths tend to converge on the same answer
    print("=" * 60)
    print("Technique 2 — Self Consistency")
    print("=" * 60)
    print(f"Generating {n_samples} reasoning paths and finding the consensus.\n")

    prompt = (
        "Watson asked: Holmes, what do you conclude from the evidence?\n"
        "Holmes replied:"
    )

    print(f"Prompt:\n{prompt}\n")

    # Generate multiple completions at higher temperature for diversity
    completions = []
    for i in range(n_samples):
        output     = generate(model, prompt, max_new_tokens=100, temperature=0.9)
        completion = output[len(prompt):]
        completions.append(completion)
        print(f"Path {i + 1}: {completion[:80]}...")

    # Majority vote — find the most common opening words across completions
    # in production you would extract the final answer token and vote on that
    # here we show the concept by finding the most common opening phrase
    first_words = [" ".join(c.split()[:5]) for c in completions]
    most_common  = Counter(first_words).most_common(1)[0][0]
    count        = Counter(first_words)[most_common]

    print(f"\nMost consistent opening: '{most_common}'")
    print(f"Appeared in {count}/{n_samples} paths\n")


# ─── Technique 3 — ReAct ─────────────────────────────────────────────────────

def react(model):
    # ReAct — Reason + Act
    # interleaves reasoning (Thought) with actions (Act) and observations (Observe)
    # the model reasons about what to do, acts, observes the result, then reasons again
    # in production ReAct connects to real tools — search engines, calculators, databases
    # here we simulate the loop with mock observations to show the structure
    # the pattern: Thought → Action → Observation → Thought → Action → ...
    print("=" * 60)
    print("Technique 3 — ReAct (Reason + Act)")
    print("=" * 60)
    print("ReAct interleaves reasoning with actions and observations.\n")

    def mock_tool(action):
        # in production this calls a real search engine, database, or API
        # the tool result becomes the next observation in the reasoning loop
        import random
        observations = [
            "The footprints suggest a man of medium height who walks with a slight limp.",
            "The letter was written hastily — the ink is smudged on the right side.",
            "The suspect was seen leaving Baker Street at half past nine.",
            "The tobacco ash is consistent with a Trichinopoly cigar.",
        ]
        return random.choice(observations)

    context = "Holmes began his investigation of the mysterious case.\n"
    print(f"Initial context:\n{context}")

    for step in range(3):
        print(f"─── Step {step + 1} ───")

        # Thought — model reasons about what to do next
        thought_prompt = context + "Thought: I should"
        thought        = generate(model, thought_prompt, max_new_tokens=50)
        thought        = thought[len(thought_prompt):]
        print(f"Thought: I should{thought[:100]}")

        # Action — model decides what to examine
        action_prompt = context + f"Thought: I should{thought}\nAction: Examine the"
        action        = generate(model, action_prompt, max_new_tokens=30)
        action        = action[len(action_prompt):]
        print(f"Action: Examine the{action[:50]}")

        # Observation — mock tool returns a result
        observation = mock_tool(action)
        print(f"Observation: {observation}")

        # Update context for next reasoning step
        context += (
            f"Thought: I should{thought[:50]}\n"
            f"Action: Examine the{action[:30]}\n"
            f"Observation: {observation}\n"
        )
        print()

    print(f"Final context:\n{context}\n")


# ─── Technique 4 — Tree of Thought ───────────────────────────────────────────

def tree_of_thought(model, branches=3, depth=2):
    # Tree of Thought — explore multiple reasoning branches at each step
    # instead of a single linear chain (CoT)
    # at each node generate multiple continuations and keep the most promising
    # in production score branches with a separate evaluator or reward model
    # here we use sentence count as a simple proxy for reasoning depth
    # the key insight: some reasoning paths are dead ends — ToT lets you backtrack
    # and explore alternatives rather than committing to one path
    print("=" * 60)
    print("Technique 4 — Tree of Thought (ToT)")
    print("=" * 60)
    print(f"Exploring {branches} branches at each of {depth} levels.\n")

    def score_branch(text):
        # Simple heuristic — count complete sentences as a proxy for coherence
        # in production use a reward model or verifier to score branches
        sentences = re.split(r'[.!?]', text)
        return len([s for s in sentences if len(s.strip()) > 10])

    root_prompt  = "Holmes examined the evidence and considered three possible explanations:\n"
    print(f"Root:\n{root_prompt}")

    current_level = [root_prompt]

    for level in range(depth):
        print(f"\n─── Level {level + 1} ───")
        next_level = []

        for node in current_level:
            node_branches = []

            for b in range(branches):
                branch = generate(model, node, max_new_tokens=80, temperature=0.9)
                branch = branch[len(node):]
                score  = score_branch(branch)
                node_branches.append((branch, score))
                print(f"Branch {b + 1} (score={score}): {branch[:60]}...")

            # Keep only the highest scoring branch at each node
            best_branch, best_score = max(node_branches, key=lambda x: x[1])
            print(f"Best branch (score={best_score}): {best_branch[:60]}...")
            next_level.append(node + best_branch)

        current_level = next_level

    print(f"\nFinal reasoning path:\n{current_level[0][:500]}\n")


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(123)

    # ── Download and prepare corpus ───────────────────────────────────────────
    text = download_corpus()
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    print(f"Vocab size:   {tokenizer.n_vocab:,}")
    print(f"Total tokens: {len(data):,}")

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
    print(f"Model parameters: {total_params:,}")

    # ── Train ─────────────────────────────────────────────────────────────────
    train(model, data, device)

    # ── Demonstrate all four reasoning techniques ─────────────────────────────
    print("\n" + "=" * 60)
    print("Reasoning Techniques")
    print("=" * 60)

    chain_of_thought(model)
    self_consistency(model, n_samples=5)
    react(model)
    tree_of_thought(model, branches=3, depth=2)