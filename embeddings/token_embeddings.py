import torch
import torch.nn as nn
import gensim.downloader as api

# Load pretrained Word2Vec model trained on Google News (3 billion words, 300 dimensions)
word_vectors = api.load("word2vec-google-news-300")

# Classic king - man + woman = queen analogy
result = word_vectors.most_similar(
    positive=["king", "woman"],
    negative=["man"],
    topn=5
)
print("Analogy results:", result)

# Semantic similarity between word pairs
pairs = [
    ("king", "queen"),
    ("man", "woman"),
    ("computer", "keyboard"),
    ("king", "computer")
]

for w1, w2 in pairs:
    sim = word_vectors.similarity(w1, w2)
    print(f"{w1} <-> {w2}: {sim:.4f}")

# Inspect the raw vector for a word
print("\nking vector:", word_vectors["king"])
print("Vector dimensions:", len(word_vectors["king"]))

# PyTorch embedding layer — randomly initialised, learned during training
torch.manual_seed(123)
vocab_size = 10
output_dim = 3
embedding_layer = nn.Embedding(vocab_size, output_dim)

print("\nEmbedding weights:\n", embedding_layer.weight)
print("Embedding for token 3:", embedding_layer(torch.tensor([3])))