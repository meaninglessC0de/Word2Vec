import numpy as np
from collections import Counter


def build_noise_distribution(word_freq: Counter, word2idx: dict[str, int])  -> np.ndarray:
    vocab_size = len(word2idx)
    noise = np.zeros(vocab_size)
    for (word, count) in word_freq.items():
        if word in word2idx:
            noise[word2idx[word]] = count ** 0.75 
    
    sum_noise = noise.sum()
    noise = noise / sum_noise
    
    return noise 

