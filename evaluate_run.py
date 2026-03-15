import torch
import evaluate
import json
embeddings = torch.load("embeddings/embeddings.pt")

with open("embeddings/word2idx.json") as f:
    word2idx = json.load(f)

with open("embeddings/idx2word.json") as f:
    idx2word = {int(k): v for k, v in json.load(f).items()}

print(evaluate.n_nearest_neighbours("apple", 5, word2idx, idx2word, embeddings))

# various rough categories of words
words = [
    "king", "queen", "prince", "princess", "throne", "crown",
    "france", "germany", "england", "italy", "spain", "russia",
    "paris", "london", "berlin", "rome", "madrid", "moscow",
    "apple", "microsoft", "google", "intel", "software", "computer",
    "dog", "cat", "horse", "bird", "fish", "animal",
    "one", "two", "three", "four", "five", "six"
]


print(evaluate.plot_tsne(words,word2idx, embeddings))