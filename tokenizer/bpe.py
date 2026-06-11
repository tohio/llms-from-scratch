import json

class CustomBPETokenizer:
    def __init__(self):
        # vocab: list of all tokens (base characters + merged tokens)
        # merges: ordered list of merge rules learned during training
        # stoi: token → ID lookup (string to index)
        # itos: ID → token lookup (index to string)
        self.vocab = []
        self.merges = []
        self.stoi = {}
        self.itos = {}

    # Before any merging, BPE starts by splitting every word into individual characters.
    def pretokenize(self, text):
        words = text.split()
        return [list(word) for word in words]

    # Counting pairs. We need to scan every word and count how often each adjacent pair appears across the whole corpus.
    def get_pairs(self, words):
        pairs = {}
        for word in words:
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                if pair in pairs:
                    pairs[pair] += 1
                else:
                    pairs[pair] = 1
        return pairs

    # Find the most frequent pair and merge it across all words
    # after a merge you don't want to accidentally re-examine the character you just merged into
    # for each word
    #     loop through positions
    #     if current + next == best_pair
    #         replace the two with the merged token
    #         skip the next position
    def merge_pair(self, words, best_pair):
        new_words = []
        for word in words:
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i + 1]) == best_pair:
                    new_word.append(word[i] + word[i + 1])
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_words.append(new_word)
        return new_words

    def train(self, text, num_merges):
        # Start from character level — vocab is all distinct characters in the corpus
        words = self.pretokenize(text)
        self.vocab = sorted(set(ch for word in words for ch in word))
        self.merges = []

        for i in range(num_merges):
            pairs = self.get_pairs(words)

            # Stop early if there are no more pairs to merge
            if not pairs:
                break

            # Pick the most frequent pair — this is the BPE merge decision
            best_pair = max(pairs, key=lambda x: pairs[x])

            # Apply the merge across all words in the corpus
            words = self.merge_pair(words, best_pair)

            # Add the new merged token to vocab and record the merge rule
            merged_token = best_pair[0] + best_pair[1]
            self.vocab.append(merged_token)
            self.merges.append(best_pair)
            print(f"Merge {i + 1}: {best_pair} → '{merged_token}'")

        # Build lookup tables after all merges are done
        self.stoi = {token: i for i, token in enumerate(self.vocab)}
        self.itos = {i: token for i, token in enumerate(self.vocab)}
        print(f"\nFinal vocab size: {len(self.vocab)}")
        print(f"Vocab: {self.vocab}")

    def encode(self, text):
        # Start from characters, then replay merge rules in the same order they were learned
        # this ensures new text is tokenized consistently with the training corpus
        words = self.pretokenize(text)
        for pair in self.merges:
            words = self.merge_pair(words, pair)

        # Flatten words into a single list of token IDs
        token_ids = []
        for word in words:
            for token in word:
                token_ids.append(self.stoi[token])
        return token_ids

    def decode(self, token_ids):
        # Convert IDs back to tokens and join with spaces
        # note: spaces between words are approximated since pretokenize splits on whitespace
        tokens = [self.itos[i] for i in token_ids]
        return " ".join(tokens)

    # Save vocab and merges to disk so we don't have to retrain every time
    def save(self, path):
        data = {
            "vocab": self.vocab,
            # merges are tuples — convert to lists for JSON compatibility
            "merges": [list(pair) for pair in self.merges]
        }
        with open(path, "w", encoding="utf8") as f:
            json.dump(data, f, indent=2)
        print(f"Tokenizer saved to {path}")

    # Load vocab and merges back, rebuild stoi and itos
    def load(self, path):
        with open(path, "r", encoding="utf8") as f:
            data = json.load(f)
        self.vocab = data["vocab"]
        # JSON loads lists — convert back to tuples so merges are consistent with training
        self.merges = [tuple(pair) for pair in data["merges"]]
        # stoi and itos are derived from vocab so no need to save them separately
        self.stoi = {token: i for i, token in enumerate(self.vocab)}
        self.itos = {i: token for i, token in enumerate(self.vocab)}
        print(f"Tokenizer loaded from {path}")


if __name__ == "__main__":
    with open("../data/tiny_corpus.txt", "r", encoding="utf8") as f:
        text = f.read()

    # Train on corpus and save
    tokenizer = CustomBPETokenizer()
    tokenizer.train(text, num_merges=50)
    tokenizer.save("../data/bpe_tokenizer.json")

    # Test encode and decode on a sample
    sample_text = "Every moment is a beginning"
    encoded = tokenizer.encode(sample_text)
    decoded = tokenizer.decode(encoded)

    print("\nOriginal:", sample_text)
    print("Encoded:", encoded)
    print("Decoded:", decoded)

    # To reuse without retraining, load from disk
    tokenizer = CustomBPETokenizer()
    tokenizer.load("../data/bpe_tokenizer.json")