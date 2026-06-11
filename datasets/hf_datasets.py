import torch
from torch.utils.data import Dataset, DataLoader
from datasets import Dataset as HFDataset
import tiktoken


class HFTinyCorpusDataset(Dataset):
    def __init__(self, path, block_size):
        # Load the raw text from disk
        with open(path, "r", encoding="utf8") as f:
            text = f.read()

        # Tokenize the entire corpus into a flat list of integer token IDs
        # using GPT-4o's BPE tokenizer
        tokenizer = tiktoken.encoding_for_model("gpt-4o")
        data      = tokenizer.encode(text)

        # Wrap the token IDs in a HuggingFace Dataset
        # HF Dataset gives us free access to map, filter, split, and save methods
        hf_dataset = HFDataset.from_dict({"input_ids": [data]})

        # Flatten the nested list into a single sequence of token IDs
        # and store as a tensor for efficient indexing
        self.data       = torch.tensor(
            hf_dataset["input_ids"][0], dtype=torch.long
        )
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
    # Instantiate the dataset with a block size of 4
    dataset = HFTinyCorpusDataset("../data/tiny_corpus.txt", block_size=4)

    # DataLoader handles batching, shuffling, and parallel loading
    # shuffle=True randomises sequence order each epoch
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    print(f"Dataset size:       {len(dataset)} sequences")
    print(f"Batches per epoch:  {len(dataloader)}")

    # Inspect the first batch to verify shapes and content
    x, y = next(iter(dataloader))
    print(f"\nInput shape:   {x.shape}")
    print(f"Target shape:  {y.shape}")
    print(f"\nInput:\n{x}")
    print(f"\nTarget:\n{y}")

    # HF Dataset bonus — show what the HF dataset object looks like
    from datasets import Dataset as HFDataset
    tokenizer  = tiktoken.encoding_for_model("gpt-4o")
    with open("../data/tiny_corpus.txt", "r", encoding="utf8") as f:
        text = f.read()
    data       = tokenizer.encode(text)
    hf_dataset = HFDataset.from_dict({"input_ids": [data]})
    print(f"\nHF Dataset:        {hf_dataset}")
    print(f"HF Dataset features: {hf_dataset.features}")