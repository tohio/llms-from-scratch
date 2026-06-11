class CharacterTokenizer:
    def __init__(self, text):
        # Build vocabulary from all distinct characters in the corpus
        # sorted ensures consistent ordering across runs
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)

        # stoi: character → ID lookup (string to index)
        # itos: ID → character lookup (index to string)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, s):
        # Convert each character to its corresponding integer ID
        return [self.stoi[c] for c in s]

    def decode(self, ids):
        # Convert each integer ID back to its character and join into a string
        return "".join([self.itos[i] for i in ids])

    def vocab_info(self):
        # Convenience method to inspect the vocabulary
        print("Vocabulary size:", self.vocab_size)
        print("Characters:", self.chars)


if __name__ == "__main__":
    with open("../data/tiny_corpus.txt", "r", encoding="utf8") as f:
        text = f.read()

    # Preview the corpus
    print(text[:200])
    print("Total characters:", len(text))

    # Build tokenizer from corpus — vocabulary is derived from the text itself
    tokenizer = CharacterTokenizer(text)
    tokenizer.vocab_info()

    # Test encode and decode on a sample
    sample_text = "Every moment is a beginning"
    encoded = tokenizer.encode(sample_text)
    decoded = tokenizer.decode(encoded)

    print("Original:", sample_text)
    print("Encoded:", encoded)
    print("Decoded:", decoded)