import re
import json
import hashlib
from datasets import load_dataset


# ─── Config ───────────────────────────────────────────────────────────────────

# Number of samples to pull from each dataset
# DPO datasets are smaller than SFT — each example needs chosen AND rejected
FINEWEB_SAMPLES  = 300
DOLMA_SAMPLES    = 500   # pull more since we'll lose some to filtering

# Output paths — prefixed by source to match pretrain_curate.py convention
FINEWEB_OUTPUT   = "../data/fineweb_dpo.jsonl"
DOLMA_OUTPUT     = "../data/dolma_dpo.jsonl"
MIXED_OUTPUT     = "../data/mixed_dpo.jsonl"

# Filtering thresholds
MIN_RESPONSE_LEN = 50    # minimum response length in characters
MAX_RESPONSE_LEN = 2000  # maximum response length in characters
MIN_PROMPT_LEN   = 10    # minimum prompt length
MIN_SCORE_GAP    = 0.5   # minimum score gap between chosen and rejected
                          # applies only to datasets with numeric scores
                          # (ultrafeedback_binarized — scored by GPT-4)
                          # hh-rlhf uses binary human preference with no
                          # numeric scores so this filter is skipped for it

# Dataset mixing ratios — must sum to 1.0
# should match pretrain_curate.py and sft_curate.py for domain consistency
MIXING_RATIOS = {
    "fineweb": 0.6,
    "dolma":   0.4
}


# ─── Load FineWeb DPO ─────────────────────────────────────────────────────────
# HuggingFaceH4/ultrafeedback_binarized is topically aligned with FineWeb-Edu
# it contains preference pairs scored by GPT-4 on educational and informational topics
# same domain as FineWeb — makes it the natural DPO complement

def load_fineweb_dpo(n_samples):
    print("Loading FineWeb DPO dataset (ultrafeedback_binarized)...")
    dataset = load_dataset(
        "HuggingFaceH4/ultrafeedback_binarized",
        split     = "train_prefs",
        streaming = True
    )

    samples = []
    for i, example in enumerate(dataset):
        if i >= n_samples:
            break

        prompt   = example.get("prompt", "").strip()
        chosen   = example.get("chosen", [])
        rejected = example.get("rejected", [])

        # extract response content from the message list
        chosen_text   = chosen[-1].get("content", "").strip()   if chosen   else ""
        rejected_text = rejected[-1].get("content", "").strip() if rejected else ""

        # extract numeric scores — used for score gap filtering
        # ultrafeedback_binarized is scored by GPT-4, scores are meaningful
        chosen_score   = example.get("score_chosen",   0)
        rejected_score = example.get("score_rejected", 0)

        if prompt and chosen_text and rejected_text:
            samples.append({
                "prompt":          prompt,
                "chosen":          chosen_text,
                "rejected":        rejected_text,
                "chosen_score":    chosen_score,
                "rejected_score":  rejected_score,
                "has_scores":      True,    # numeric scores available
                "source":          "fineweb_dpo"
            })

    print(f"Loaded {len(samples)} FineWeb DPO samples")
    return samples


# ─── Load Dolma DPO ───────────────────────────────────────────────────────────
# Anthropic/hh-rlhf is topically aligned with Dolma
# it contains human preference pairs on helpful and harmless responses
# diverse domain coverage matches Dolma's web text diversity
# note: hh-rlhf uses binary human preference — no numeric scores
# the score gap filter is skipped for this source

def load_dolma_dpo(n_samples):
    print("\nLoading Dolma DPO dataset (hh-rlhf)...")
    dataset = load_dataset(
        "Anthropic/hh-rlhf",
        split     = "train",
        streaming = True
    )

    samples = []
    for i, example in enumerate(dataset):
        if i >= n_samples:
            break

        chosen_text   = example.get("chosen",   "").strip()
        rejected_text = example.get("rejected", "").strip()

        # parse the conversation — last Human turn is prompt
        # last Assistant turn is the response
        def parse_conversation(text):
            parts = text.split("\n\nHuman:")
            if len(parts) < 2:
                return "", ""
            last_turn  = parts[-1]
            human_part = last_turn.split("\n\nAssistant:")
            prompt     = human_part[0].strip()
            response   = human_part[-1].strip() if len(human_part) > 1 else ""
            return prompt, response

        prompt,   chosen_response   = parse_conversation(chosen_text)
        _,        rejected_response = parse_conversation(rejected_text)

        if prompt and chosen_response and rejected_response:
            samples.append({
                "prompt":          prompt,
                "chosen":          chosen_response,
                "rejected":        rejected_response,
                "chosen_score":    None,   # hh-rlhf has no numeric scores
                "rejected_score":  None,   # binary human preference only
                "has_scores":      False,  # score gap filter skipped
                "source":          "dolma_dpo"
            })

    print(f"Loaded {len(samples)} Dolma DPO samples")
    return samples


# ─── Filter ───────────────────────────────────────────────────────────────────

def filter_samples(samples):
    print("\nFiltering...")
    filtered = []

    for example in samples:
        prompt   = example["prompt"]
        chosen   = example["chosen"]
        rejected = example["rejected"]

        # prompt length filter
        if len(prompt) < MIN_PROMPT_LEN:
            continue

        # response length filter — both chosen and rejected
        if len(chosen) < MIN_RESPONSE_LEN or len(chosen) > MAX_RESPONSE_LEN:
            continue
        if len(rejected) < MIN_RESPONSE_LEN or len(rejected) > MAX_RESPONSE_LEN:
            continue

        # score gap filter — only applied when numeric scores are available
        # hh-rlhf uses binary human preference with no numeric scores
        # so the filter is skipped for that source to avoid a no-op comparison
        if example["has_scores"]:
            score_gap = example["chosen_score"] - example["rejected_score"]
            if score_gap < MIN_SCORE_GAP:
                continue

        # chosen and rejected should not be identical
        if chosen == rejected:
            continue

        # punctuation filter
        if not re.search(r'[.!?]', chosen):
            continue
        if not re.search(r'[.!?]', rejected):
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
        # fingerprint based on prompt — avoid duplicate questions
        fingerprint = hashlib.md5(
            example["prompt"][:200].encode("utf8")
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
        prompt   = example["prompt"]
        chosen   = example["chosen"]
        rejected = example["rejected"]

        # remove HTML tags
        prompt   = re.sub(r'<[^>]+>', '', prompt)
        chosen   = re.sub(r'<[^>]+>', '', chosen)
        rejected = re.sub(r'<[^>]+>', '', rejected)

        # remove URLs
        prompt   = re.sub(r'http\S+|www\S+', '', prompt)
        chosen   = re.sub(r'http\S+|www\S+', '', chosen)
        rejected = re.sub(r'http\S+|www\S+', '', rejected)

        # normalize whitespace
        prompt   = re.sub(r'\s+', ' ', prompt).strip()
        chosen   = re.sub(r'\s+', ' ', chosen).strip()
        rejected = re.sub(r'\s+', ' ', rejected).strip()

        # skip if cleaning left too little text
        if len(prompt) < MIN_PROMPT_LEN:
            continue
        if len(chosen) < MIN_RESPONSE_LEN or len(rejected) < MIN_RESPONSE_LEN:
            continue

        cleaned.append({
            "prompt":    prompt,
            "chosen":    chosen,
            "rejected":  rejected,
            "source":    example["source"]
        })

    print(f"Kept {len(cleaned)} / {len(samples)} samples after cleaning")
    return cleaned


# ─── Mix ──────────────────────────────────────────────────────────────────────

def mix_datasets(datasets):
    # Combine FineWeb and Dolma DPO datasets using configured mixing ratios
    # ratios match pretrain_curate.py and sft_curate.py for domain consistency
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
    # Save as jsonl — one prompt/chosen/rejected triplet per line
    # compatible with dpo.py's DPODataset class
    with open(path, "w", encoding="utf8") as f:
        for example in samples:
            f.write(json.dumps({
                "prompt":   example["prompt"],
                "chosen":   example["chosen"],
                "rejected": example["rejected"]
            }) + "\n")
    print(f"Saved {len(samples)} examples to {path}")


# ─── Pipeline ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── FineWeb DPO ───────────────────────────────────────────────────────────
    print("=" * 60)
    print("FineWeb DPO — ultrafeedback_binarized")
    print("=" * 60)

    fineweb_samples = load_fineweb_dpo(FINEWEB_SAMPLES)
    fineweb_samples = filter_samples(fineweb_samples)
    fineweb_samples = deduplicate(fineweb_samples)
    fineweb_samples = clean_samples(fineweb_samples)
    save(fineweb_samples, FINEWEB_OUTPUT)

    print(f"\nFineWeb DPO stats:")
    print(f"  Examples:         {len(fineweb_samples)}")
    print(f"  Avg prompt len:   {sum(len(s['prompt'])   for s in fineweb_samples) // max(len(fineweb_samples), 1):,} chars")
    print(f"  Avg chosen len:   {sum(len(s['chosen'])   for s in fineweb_samples) // max(len(fineweb_samples), 1):,} chars")
    print(f"  Avg rejected len: {sum(len(s['rejected']) for s in fineweb_samples) // max(len(fineweb_samples), 1):,} chars")

    # ── Dolma DPO ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Dolma DPO — hh-rlhf")
    print("=" * 60)

    dolma_samples = load_dolma_dpo(DOLMA_SAMPLES)
    dolma_samples = filter_samples(dolma_samples)
    dolma_samples = deduplicate(dolma_samples)
    dolma_samples = clean_samples(dolma_samples)
    save(dolma_samples, DOLMA_OUTPUT)

    print(f"\nDolma DPO stats:")
    print(f"  Examples:         {len(dolma_samples)}")
    print(f"  Avg prompt len:   {sum(len(s['prompt'])   for s in dolma_samples) // max(len(dolma_samples), 1):,} chars")
    print(f"  Avg chosen len:   {sum(len(s['chosen'])   for s in dolma_samples) // max(len(dolma_samples), 1):,} chars")
    print(f"  Avg rejected len: {sum(len(s['rejected']) for s in dolma_samples) // max(len(dolma_samples), 1):,} chars")

    # ── Mixed DPO ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Mixing DPO datasets")
    print("=" * 60)

    mixed_samples = mix_datasets({
        "fineweb": fineweb_samples,
        "dolma":   dolma_samples
    })
    save(mixed_samples, MIXED_OUTPUT)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("DPO curation complete")
    print("=" * 60)
    print(f"  FineWeb DPO → {FINEWEB_OUTPUT}")
    print(f"  Dolma DPO   → {DOLMA_OUTPUT}")
    print(f"  Mixed DPO   → {MIXED_OUTPUT}")
    print("\nAll DPO datasets are ready for dpo.py")