import os
import re
import random
import hashlib
import urllib.request
from datasets import load_dataset
from langdetect import detect, LangDetectException


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
LANG_THRESHOLD   = 0.8    # minimum confidence for language detection

# Quality scoring thresholds — heuristic based
# these catch low-quality documents without loading a model
# alternative: perplexity scoring using a small model (e.g. GPT-2) is more
# accurate but requires loading a model and is significantly slower
MAX_DIGIT_RATIO  = 0.2    # documents with >20% digits are likely tables/code
MAX_UPPER_RATIO  = 0.3    # documents with >30% uppercase are likely spam/ads
MAX_SYMBOL_RATIO = 0.1    # documents with >10% symbols are likely garbled
MIN_PUNCT_RATIO  = 0.01   # documents with <1% punctuation lack sentence structure

# Dataset mixing ratios — must sum to 1.0
# adjust these to control the proportion of each source in the final corpus
# production models like LLaMA and Falcon use carefully tuned mixing ratios
MIXING_RATIOS = {
    "fineweb": 0.6,   # 60% FineWeb-Edu — high quality educational content
    "dolma":   0.4    # 40% Dolma — diverse web text
}


# ─── Language Detection ───────────────────────────────────────────────────────
# langdetect — pure Python, no NumPy dependency, no model download needed
# slightly slower than fasttext but accurate enough for web text filtering
# and avoids the NumPy 2.x incompatibility that breaks fasttext

def is_english(text):
    try:
        return detect(text) == "en"
    except LangDetectException:
        # langdetect raises LangDetectException on very short or garbled text
        # treat as non-English to skip
        return False


def filter_language(samples):
    print("\nFiltering by language...")
    english = [s for s in samples if is_english(s)]
    print(f"Kept {len(english)} / {len(samples)} English samples")
    return english


# ─── Quality Scoring ──────────────────────────────────────────────────────────
# Heuristic quality scoring — measures surface properties of text that
# correlate with quality without requiring a model
# catches spam, tables, code dumps, and garbled text
# alternative: perplexity scoring using GPT-2 via transformers is more
# principled but requires a model load and is ~100x slower per document

def quality_score(text):
    # Count character types as ratios of total length
    chars      = len(text)
    digits     = sum(c.isdigit()  for c in text) / chars
    upper      = sum(c.isupper()  for c in text) / chars
    symbols    = sum(not c.isalnum() and not c.isspace() for c in text) / chars
    punct      = sum(c in ".!?," for c in text) / chars

    # All ratios must be within bounds — any failure disqualifies the document
    if digits  > MAX_DIGIT_RATIO:  return False
    if upper   > MAX_UPPER_RATIO:  return False
    if symbols > MAX_SYMBOL_RATIO: return False
    if punct   < MIN_PUNCT_RATIO:  return False
    return True


def filter_quality(samples):
    print("\nFiltering by quality...")
    filtered = [s for s in samples if quality_score(s)]
    print(f"Kept {len(filtered)} / {len(samples)} samples after quality filter")
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

    min_size = min(len(samples) / MIXING_RATIOS[name] for name, samples in datasets.items())

    mixed = []
    for name, samples in datasets.items():
        n = int(min_size * MIXING_RATIOS[name])
        mixed.extend(samples[:n])
        print(f"  {name}: {n} samples ({MIXING_RATIOS[name]*100:.0f}%)")

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
    # allenai/dolma requires a dataset script which is no longer supported
    # by newer versions of the datasets library.
    # allenai/dolma-pes2o is a subset of Dolma (peS2o — scientific papers)
    # that loads as a standard parquet dataset with no script required
    # it is diverse, high quality web text — a good raw source for curation
    dataset = load_dataset(
        "allenai/dolma-pes2o",
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

    # ── FineWeb-Edu — already curated, light touch pipeline ──────────────────
    print("\n" + "=" * 60)
    print("FineWeb-Edu — clean curated source")
    print("=" * 60)

    fineweb_samples = load_fineweb(FINEWEB_SAMPLES)
    fineweb_samples = filter_language(fineweb_samples)
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
    dolma_samples = filter_language(dolma_samples)
    dolma_samples = filter_samples(dolma_samples)
    dolma_samples = filter_quality(dolma_samples)
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