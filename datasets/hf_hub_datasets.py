import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import tiktoken


class HFHubDataset(Dataset):
    def __init__(self, dataset_name, block_size, split="train", text_column="text"):
        # Pull dataset directly from HuggingFace Hub
        # dataset_name: e.g. "roneneldan/TinyStories" or "wikitext"
        # split: "train", "validation", or "test"
        print(f"Loading {dataset_name} ({split}) from HuggingFace Hub...")
        hf_dataset = load_dataset(dataset_name, split=split)

        # Concatenate all text samples into one long string
        # some datasets have multiple rows — we join them into a single corpus
        full_text = " ".join(hf_dataset[text_column])

        # Tokenize the entire corpus into a flat list of integer token IDs
        # using GPT-4o's BPE tokenizer
        tokenizer = tiktoken.encoding_for_model("gpt-4o")
        data      = tokenizer.encode(full_text)

        print(f"Total tokens: {len(data):,}")

        # Store as a tensor for efficient indexing during training
        self.data       = torch.tensor(data, dtype=torch.long)
        self.block_size = block_size

    def __len__(self):
        # Total number of valid sequences we can extract from the corpus
        # each sequence needs block_size tokens for x and block_size tokens for y
        # so the last valid start index is len(data) - block_size
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        # x is the input sequence — block_size tokens starting at idx
        # y is the target sequence — same length but shifted one position right
        # the model learns to predict y[t] given x[0..t]
        x = self.data[idx:idx + self.block_size]
        y = self.data[idx + 1:idx + self.block_size + 1]
        return x, y


if __name__ == "__main__":
    # TinyStories — a small dataset of short children's stories
    # good for testing since it's small and downloads quickly
    # text_column is "text" for most HF text datasets
    dataset = HFHubDataset(
        dataset_name="roneneldan/TinyStories",
        block_size=64,     # larger block size since we have more data
        split="train",
        text_column="text"
    )

    # DataLoader handles batching, shuffling, and parallel loading
    # shuffle=True randomises sequence order each epoch
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    print(f"\nDataset size:       {len(dataset):,} sequences")
    print(f"Batches per epoch:  {len(dataloader):,}")

    # Inspect the first batch to verify shapes and content
    x, y = next(iter(dataloader))
    print(f"\nInput shape:   {x.shape}")
    print(f"Target shape:  {y.shape}")
    print(f"\nInput:\n{x}")
    print(f"\nTarget:\n{y}")