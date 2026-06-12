import re
import hashlib
import urllib.request
from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import fasttext


# ─── Config ───────────────────────────────────────────────────────────────────

# Number of samples to pull from each dataset
# keep small for demonstration purposes
FINEWEB_SAMPLES  = 1000
DOLMA_SAMPLES    = 2000   # pull more since we'll lose some to filtering

# Output paths
FINEWEB_OUTPUT   = "../data/fineweb_corpus.txt"
DOLMA_OUTPUT     = "../data/dolma_corpus.txt"
MIXED_OUTPUT     = "../data/mixed_corpus.txt"

# Filtering thresholds
MIN_LENGTH       = 100    # minimum character length
MAX_LENGTH       = 10000  # maximum character length
MIN_WORDS        = 20     # minimum word count
MAX_WORD_RATIO   = 0.5    # maximum ratio of long words (filters garbled text)
MAX_PERPLEXITY   = 1000   # documents above this perplexity are low quality
LANG_THRESHOLD   = 0.8    # minimum confidence for language detection

# Dataset mixing ratios — must sum to 1.0
# adjust these to control the proportion of each source in the final corpus
# production models like LLaMA and Falcon use carefully tuned mixing ratios
MIXING_RATIOS    = {
    "fineweb": 0.6,   # 60% FineWeb-Edu — high quality educational content
    "dolma":   0.4    # 40% Dolma — diverse web text
}

# FastText language model path
FASTTEXT_MODEL_PATH = "lid.176.ftz"
FASTTEXT_MODEL_URL  = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"

# Perplexity model — GPT-2 is small enough to run locally on CPU
# alternative: use heuristic scoring (punctuation density, digit ratio,
# uppercase ratio, symbol ratio) if you want to avoid loading a model entirely
# heuristic is faster but less accurate than perplexity based filtering
PERPLEXITY_MODEL = "gpt2"


# ─── Language Detection ───────────────────────────────────────────────────────

def load_fasttext_model():
    # Download the fasttext language identification model if not present
    # lid.176.ftz is a compressed model that identifies 176 languages
    # only 900KB — runs fully locally, no GPU needed
    import os
    if not os.path.exists(FASTTEXT_MODEL_PATH):
        print(f"Downloading fasttext model...")
        urllib.request.urlretrieve(FASTTEXT_MODEL_URL, FASTTEXT_MODEL_PATH)
    return fasttext.load_model(FASTTEXT_MODEL_PATH)


def is_english(text, model, threshold=LANG_THRESHOLD):
    # Predict language — fasttext returns label and confidence score
    # replace newlines since fasttext treats them as document separators
    label, score = model.predict(text.replace("\n", " "))
    return label[0] == "__label__en" and score[0] >= threshold


def filter_language(samples, model):
    print("\nFiltering by language...")
    english = [s for s in samples if is_english(s, model)]
    print(f"Kept {len(english)} / {len(samples)} English samples")
    return english


# ─── Perplexity Scoring ───────────────────────────────────────────────────────

def load_perplexity_model():
    # GPT-2 is used as the scoring model — small enough to run on CPU
    # alternative: replace with a heuristic scorer that measures punctuation
    # density, digit ratio, uppercase ratio, and symbol ratio — much faster
    # but less accurate than perplexity based filtering
    print("\nLoading perplexity model (GPT-2)...")
    device    = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(PERPLEXITY_MODEL)
    model     = AutoModelForCausalLM.from_pretrained(PERPLEXITY_MODEL).to(device)
    model.eval()
    return tokenizer, model, device


def compute_perplexity(text, tokenizer, model, device):
    # Perplexity measures how surprised the model is by the text
    # low perplexity = natural, coherent text the model finds predictable
    # high perplexity = garbled, unnatural, or out-of-distribution text
    inputs = tokenizer(
        text[:512],              # truncate to 512 tokens for speed
        return_tensors  = "pt",
        truncation      = True,
        max_length      = 512
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        # cross entropy loss — exponentiate to get perplexity
        perplexity = torch.exp(outputs.loss).item()

    return perplexity


def filter_perplexity(samples, tokenizer, model, device):
    print("\nFiltering by perplexity...")
    filtered = []

    for text in samples:
        ppl = compute_perplexity(text, tokenizer, model, device)
        if ppl <= MAX_PERPLEXITY:
            filtered.append(text)

    print(f"Kept {len(filtered)} / {len(samples)} samples after perplexity filter")
    return filtered


# ─── Filter ───────────────────────────────────────────────────────────────────

def filter_samples(samples):
    print("\nFiltering...")
    filtered = []

    for text in samples:
        # length filter — remove too short or too long documents
        if len(text) < MIN_LENGTH or len(text) > MAX_LENGTH:
            continue

        # word count filter — remove documents with too few words
        words = text.split()
        if len(words) < MIN_WORDS:
            continue

        # long word ratio filter — high ratio suggests garbled or encoded text
        # legitimate English text rarely has many very long tokens
        long_words = [w for w in words if len(w) > 20]
        if len(long_words) / len(words) > MAX_WORD_RATIO:
            continue

        # punctuation filter — remove documents with no sentence ending punctuation
        # legitimate text always has periods, exclamation marks or question marks
        if not re.search(r'[.!?]', text):
            continue

        filtered.append(text)

    print(f"Kept {len(filtered)} / {len(samples)} samples after filtering")
    return filtered


# ─── Deduplicate ──────────────────────────────────────────────────────────────

def deduplicate_exact(samples):
    # Exact deduplication using MD5 hashing
    # two documents with identical content produce the same hash
    print("\nExact deduplication...")
    seen   = set()
    unique = []

    for text in samples:
        # hash the first 500 characters — faster than hashing the full document
        fingerprint = hashlib.md5(text[:500].encode("utf8")).hexdigest()
        if fingerprint not in seen:
            seen.add(fingerprint)
            unique.append(text)

    print(f"Kept {len(unique)} / {len(samples)} samples after exact deduplication")
    return unique


def deduplicate_substring(samples):
    # Substring deduplication — remove documents that are substrings of others
    # this catches cases where one document is fully contained within another
    # O(n²) complexity — only practical on small datasets
    # production pipelines use suffix arrays for efficiency at scale
    print("\nSubstring deduplication...")
    unique = []

    for i, text in enumerate(samples):
        is_substring = False
        for j, other in enumerate(samples):
            if i != j and text in other:
                is_substring = True
                break
        if not is_substring:
            unique.append(text)

    print(f"Kept {len(unique)} / {len(samples)} samples after substring deduplication")
    return unique


# ─── Clean ────────────────────────────────────────────────────────────────────

def clean_samples(samples):
    print("\nCleaning...")
    cleaned = []

    for text in samples:
        # remove HTML tags — common in web scraped text
        text = re.sub(r'<[^>]+>', '', text)

        # remove URLs — not useful for language modeling
        text = re.sub(r'http\S+|www\S+', '', text)

        # normalize whitespace — collapse multiple spaces and newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)

        # strip leading and trailing whitespace
        text = text.strip()

        # skip if cleaning left us with too little text
        if len(text) < MIN_LENGTH:
            continue

        cleaned.append(text)

    print(f"Kept {len(cleaned)} / {len(samples)} samples after cleaning")
    return cleaned


# ─── Dataset Mixing ───────────────────────────────────────────────────────────

def mix_datasets(datasets):
    # Combine multiple curated sources using the configured mixing ratios
    # mixing ratios control how much each source contributes to the final corpus
    # this is one of the most impactful decisions in pretraining data preparation
    # frontier labs treat their mixing ratios as trade secrets
    print("\nMixing datasets...")

    # find the total target size based on the smallest proportional dataset
    sizes      = {name: len(samples) for name, samples in datasets.items()}
    min_ratio  = min(MIXING_RATIOS.values())
    min_size   = min(sizes[name] / MIXING_RATIOS[name] for name in datasets)

    mixed = []
    for name, samples in datasets.items():
        # take the proportion of each dataset as specified in MIXING_RATIOS
        n = int(min_size * MIXING_RATIOS[name])
        mixed.extend(samples[:n])
        print(f"  {name}: {n} samples ({MIXING_RATIOS[name]*100:.0f}%)")

    # shuffle to interleave sources
    import random
    random.shuffle(mixed)

    print(f"Total mixed samples: {len(mixed)}")
    return mixed


# ─── Load FineWeb-Edu ─────────────────────────────────────────────────────────

def load_fineweb(n_samples):
    print("Loading FineWeb-Edu...")
    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name      = "sample-10BT",
        split     = "train",
        streaming = True
    )

    samples = []
    for i, example in enumerate(dataset):
        if i >= n_samples:
            break
        samples.append(example["text"])

    print(f"Loaded {len(samples)} FineWeb-Edu samples")
    return samples


# ─── Load Dolma ───────────────────────────────────────────────────────────────

def load_dolma(n_samples):
    print("\nLoading Dolma...")
    dataset = load_dataset(
        "allenai/dolma",
        name      = "v1_6-sample",
        split     = "train",
        streaming = True
    )

    samples = []
    for i, example in enumerate(dataset):
        if i >= n_samples:
            break
        samples.append(example["text"])

    print(f"Loaded {len(samples)} Dolma samples")
    return samples


# ─── Save ─────────────────────────────────────────────────────────────────────

def save(samples, path):
    with open(path, "w", encoding="utf8") as f:
        f.write("\n\n".join(samples))
    print(f"Saved to {path}")


# ─── Pipeline ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # Load shared models once — reused across both datasets
    lang_model                        = load_fasttext_model()
    ppl_tokenizer, ppl_model, device  = load_perplexity_model()

    # ── FineWeb-Edu — already curated, light touch pipeline ──────────────────
    print("\n" + "=" * 60)
    print("FineWeb-Edu — clean curated source")
    print("=" * 60)

    fineweb_samples = load_fineweb(FINEWEB_SAMPLES)
    fineweb_samples = filter_language(fineweb_samples, lang_model)
    fineweb_samples = deduplicate_exact(fineweb_samples)
    fineweb_samples = clean_samples(fineweb_samples)
    save(fineweb_samples, FINEWEB_OUTPUT)

    print(f"\nFineWeb-Edu stats:")
    print(f"  Samples:     {len(fineweb_samples)}")
    print(f"  Total chars: {sum(len(s) for s in fineweb_samples):,}")
    print(f"  Avg length:  {sum(len(s) for s in fineweb_samples) // len(fineweb_samples):,} chars")

    # ── Dolma — full curation pipeline ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("Dolma — raw source with full curation pipeline")
    print("=" * 60)

    dolma_samples = load_dolma(DOLMA_SAMPLES)
    dolma_samples = filter_language(dolma_samples, lang_model)
    dolma_samples = filter_samples(dolma_samples)
    dolma_samples = filter_perplexity(dolma_samples, ppl_tokenizer, ppl_model, device)
    dolma_samples = deduplicate_exact(dolma_samples)
    dolma_samples = deduplicate_substring(dolma_samples)
    dolma_samples = clean_samples(dolma_samples)
    save(dolma_samples, DOLMA_OUTPUT)

    print(f"\nDolma stats:")
    print(f"  Samples:     {len(dolma_samples)}")
    print(f"  Total chars: {sum(len(s) for s in dolma_samples):,}")
    print(f"  Avg length:  {sum(len(s) for s in dolma_samples) // len(dolma_samples):,} chars")

    # ── Mix — combine both sources using configured ratios ────────────────────
    print("\n" + "=" * 60)
    print("Mixing datasets")
    print("=" * 60)

    mixed_samples = mix_datasets({
        "fineweb": fineweb_samples,
        "dolma":   dolma_samples
    })
    save(mixed_samples, MIXED_OUTPUT)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Curation complete")
    print("=" * 60)
    print(f"  FineWeb-Edu → {FINEWEB_OUTPUT}")
    print(f"  Dolma       → {DOLMA_OUTPUT}")
    print(f"  Mixed       → {MIXED_OUTPUT}")
    print("\nAll corpora are ready for tokenization and training.")