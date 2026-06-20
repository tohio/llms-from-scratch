import re
import json
import hashlib
from datasets import load_dataset


# ─── Config ───────────────────────────────────────────────────────────────────

# Number of samples to pull from each dataset
# SFT datasets are smaller than pretraining — quality over quantity
FINEWEB_SAMPLES  = 500
DOLMA_SAMPLES    = 800   # pull more since we'll lose some to filtering

# Output paths — prefixed by source to match pretrain_curate.py convention
FINEWEB_OUTPUT   = "../data/fineweb_sft.jsonl"
DOLMA_OUTPUT     = "../data/dolma_sft.jsonl"
MIXED_OUTPUT     = "../data/mixed_sft.jsonl"

# Filtering thresholds
MIN_RESPONSE_LEN = 50    # minimum response length in characters
MAX_RESPONSE_LEN = 2000  # maximum response length in characters
MIN_PROMPT_LEN   = 10    # minimum prompt/instruction length

# Dataset mixing ratios — must sum to 1.0
MIXING_RATIOS = {
    "fineweb": 0.6,
    "dolma":   0.4
}


# ─── Load FineWeb SFT ─────────────────────────────────────────────────────────
# HuggingFaceH4/ultrachat_200k is topically aligned with FineWeb-Edu
# it contains multi-turn conversations on educational and informational topics
# same domain as FineWeb — makes it the natural SFT complement

def load_fineweb_sft(n_samples):
    print("Loading FineWeb SFT dataset (ultrachat_200k)...")
    sft_dataset = load_dataset(
        "HuggingFaceH4/ultrachat_200k",
        split     = "train_sft",
        streaming = True
    )

    samples = []
    for i, example in enumerate(sft_dataset):
        if i >= n_samples:
            break

        # ultrachat stores conversations as a list of messages
        # extract the first user message as instruction
        # and the first assistant message as response
        messages = example.get("messages", [])
        if len(messages) < 2:
            continue

        instruction = messages[0].get("content", "").strip()
        response    = messages[1].get("content", "").strip()

        if instruction and response:
            samples.append({
                "instruction": instruction,
                "response":    response,
                "source":      "fineweb_sft"
            })

    print(f"Loaded {len(samples)} FineWeb SFT samples")
    return samples


# ─── Load Dolma SFT ───────────────────────────────────────────────────────────
# allenai/tulu-3-sft-mixture is topically aligned with Dolma
# both are from AI2 — same organization, same domain philosophy
# diverse mix of instruction following data matching Dolma's web diversity

def load_dolma_sft(n_samples):
    print("\nLoading Dolma SFT dataset (tulu-3-sft-mixture)...")
    dataset = load_dataset(
        "allenai/tulu-3-sft-mixture",
        split     = "train",
        streaming = True
    )

    samples = []
    for i, example in enumerate(dataset):
        if i >= n_samples:
            break

        # tulu stores conversations as a list of messages
        messages = example.get("messages", [])
        if len(messages) < 2:
            continue

        instruction = messages[0].get("content", "").strip()
        response    = messages[1].get("content", "").strip()

        if instruction and response:
            samples.append({
                "instruction": instruction,
                "response":    response,
                "source":      "dolma_sft"
            })

    print(f"Loaded {len(samples)} Dolma SFT samples")
    return samples


# ─── Filter ───────────────────────────────────────────────────────────────────

def filter_samples(samples):
    print("\nFiltering...")
    filtered = []

    for example in samples:
        instruction = example["instruction"]
        response    = example["response"]

        # instruction length filter — too short instructions are low quality
        if len(instruction) < MIN_PROMPT_LEN:
            continue

        # response length filter — too short or too long responses
        if len(response) < MIN_RESPONSE_LEN or len(response) > MAX_RESPONSE_LEN:
            continue

        # punctuation filter — response should have sentence ending punctuation
        if not re.search(r'[.!?]', response):
            continue

        # URL filter — remove examples where response is mostly URLs
        url_count = len(re.findall(r'http\S+|www\S+', response))
        if url_count > 3:
            continue

        filtered.append(example)

    print(f"Kept {len(filtered)} / {len(samples)} samples after filtering")
    return filtered


# ─── Deduplicate ──────────────────────────────────────────────────────────────

def deduplicate(samples):
    print("\nDeduplicating...")
    seen   = set()
    unique = []

    for example in samples:
        # fingerprint based on instruction — avoid duplicate questions
        fingerprint = hashlib.md5(
            example["instruction"][:200].encode("utf8")
        ).hexdigest()

        if fingerprint not in seen:
            seen.add(fingerprint)
            unique.append(example)

    print(f"Kept {len(unique)} / {len(samples)} samples after deduplication")
    return unique


# ─── Clean ────────────────────────────────────────────────────────────────────

def clean_samples(samples):
    print("\nCleaning...")
    cleaned = []

    for example in samples:
        instruction = example["instruction"]
        response    = example["response"]

        # remove HTML tags
        instruction = re.sub(r'<[^>]+>', '', instruction)
        response    = re.sub(r'<[^>]+>', '', response)

        # remove URLs
        instruction = re.sub(r'http\S+|www\S+', '', instruction)
        response    = re.sub(r'http\S+|www\S+', '', response)

        # normalize whitespace
        instruction = re.sub(r'\s+', ' ', instruction).strip()
        response    = re.sub(r'\s+', ' ', response).strip()

        # skip if cleaning left too little text
        if len(instruction) < MIN_PROMPT_LEN or len(response) < MIN_RESPONSE_LEN:
            continue

        cleaned.append({
            "instruction": instruction,
            "response":    response,
            "source":      example["source"]
        })

    print(f"Kept {len(cleaned)} / {len(samples)} samples after cleaning")
    return cleaned


# ─── Mix ──────────────────────────────────────────────────────────────────────

def mix_datasets(datasets):
    # Combine FineWeb and Dolma SFT datasets using configured mixing ratios
    # ratios should match pretrain_curate.py for domain consistency
    print("\nMixing datasets...")

    import random
    mixed    = []
    min_size = min(len(samples) / MIXING_RATIOS[name] for name, samples in datasets.items())

    for name, samples in datasets.items():
        n = int(min_size * MIXING_RATIOS[name])
        mixed.extend(samples[:n])
        print(f"  {name}: {n} samples ({MIXING_RATIOS[name]*100:.0f}%)")

    random.shuffle(mixed)
    print(f"Total mixed samples: {len(mixed)}")
    return mixed


# ─── Save ─────────────────────────────────────────────────────────────────────

def save(samples, path):
    # Save as jsonl — one instruction/response pair per line
    # compatible with sft.py's SFTDataset class
    with open(path, "w", encoding="utf8") as f:
        for example in samples:
            f.write(json.dumps({
                "instruction": example["instruction"],
                "response":    example["response"]
            }) + "\n")
    print(f"Saved {len(samples)} examples to {path}")


# ─── Pipeline ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── FineWeb SFT ───────────────────────────────────────────────────────────
    print("=" * 60)
    print("FineWeb SFT — ultrachat_200k")
    print("=" * 60)

    fineweb_samples = load_fineweb_sft(FINEWEB_SAMPLES)
    fineweb_samples = filter_samples(fineweb_samples)
    fineweb_samples = deduplicate(fineweb_samples)
    fineweb_samples = clean_samples(fineweb_samples)
    save(fineweb_samples, FINEWEB_OUTPUT)

    print(f"\nFineWeb SFT stats:")
    print(f"  Examples:       {len(fineweb_samples)}")
    print(f"  Avg prompt len: {sum(len(s['instruction']) for s in fineweb_samples) // len(fineweb_samples):,} chars")
    print(f"  Avg resp len:   {sum(len(s['response']) for s in fineweb_samples) // len(fineweb_samples):,} chars")

    # ── Dolma SFT ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Dolma SFT — tulu-3-sft-mixture")
    print("=" * 60)

    dolma_samples = load_dolma_sft(DOLMA_SAMPLES)
    dolma_samples = filter_samples(dolma_samples)
    dolma_samples = deduplicate(dolma_samples)
    dolma_samples = clean_samples(dolma_samples)
    save(dolma_samples, DOLMA_OUTPUT)

    print(f"\nDolma SFT stats:")
    print(f"  Examples:       {len(dolma_samples)}")
    print(f"  Avg prompt len: {sum(len(s['instruction']) for s in dolma_samples) // len(dolma_samples):,} chars")
    print(f"  Avg resp len:   {sum(len(s['response']) for s in dolma_samples) // len(dolma_samples):,} chars")

    # ── Mixed SFT ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Mixing SFT datasets")
    print("=" * 60)

    mixed_samples = mix_datasets({
        "fineweb": fineweb_samples,
        "dolma":   dolma_samples
    })
    save(mixed_samples, MIXED_OUTPUT)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SFT curation complete")
    print("=" * 60)
    print(f"  FineWeb SFT → {FINEWEB_OUTPUT}")
    print(f"  Dolma SFT   → {DOLMA_OUTPUT}")
    print(f"  Mixed SFT   → {MIXED_OUTPUT}")
    print("\nAll SFT datasets are ready for sft.py")