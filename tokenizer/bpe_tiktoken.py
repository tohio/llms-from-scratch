import tiktoken

class BPETokenizer:
    def __init__(self, model="gpt-4o"):
        # Load the tiktoken encoding for the specified model
        # different models use different tokenizers e.g. gpt-4o vs gpt-3.5-turbo
        self.tokenizer = tiktoken.encoding_for_model(model)

    def encode(self, text):
        # Convert text to a list of integer token IDs
        return self.tokenizer.encode(text)

    def decode(self, token_ids):
        # Convert token IDs back to a string
        return self.tokenizer.decode(token_ids)

    def inspect(self, text):
        # Convenience method to see the full tokenization breakdown
        token_ids = self.encode(text)
        print(f"Text:          {text}")
        print(f"Token IDs:     {token_ids}")
        print(f"Token count:   {len(token_ids)}")
        print(f"Decoded:       {self.decode(token_ids)}")


if __name__ == "__main__":
    tokenizer = BPETokenizer()

    sample_text = "Every moment is a beginning"

    # Full breakdown
    tokenizer.inspect(sample_text)