import os
from collections import Counter
from pathlib import Path

import operator
import math
import random 
import torch 

# helper functions for dataset preprocessing

def load_text8(file_path):
    if not os.path.exists(file_path):
           print("This file does not exist!")
    file = open(file_path)
    text = file.read().strip()
    tokens = text.split()
    file.close()
    return tokens 

def compute_word_frequencies(tokens):
    return Counter(tokens)
    

def filter_rare_words(tokens, freq_table, min_freq):
    filtered_tokens = []
    for token in tokens:
        if freq_table[token] >= min_freq:
            filtered_tokens.append(token)
    
    return filtered_tokens.filter

def build_vocabulary(tokens, min_count, vocab_size):
    
    word_freq = Counter(tokens)
    top_words = [word for word, count in word_freq.most_common(vocab_size) if count >= min_count]
    

    word2idx = {word: i  for i, word in enumerate(top_words)}
    idx2word = {i: word  for i, word in enumerate(top_words)}
    
    corpus = []
    for token in tokens:
        if token in word2idx:
            corpus.append(word2idx[token])
            
    return word2idx, idx2word, corpus, word_freq
        
# n.b., subsampling frequent here words improves the quality of the context we use.
    
def subsample(corpus: list[int], word_freq: Counter, word2idx: dict[str, int], threshold: float)  -> list[int]:
    total_tokens = sum(word_freq.values())
    freq = {}
    for (word,count) in word_freq.items():
        if word in word2idx:
            freq[word2idx[word]] = count / total_tokens
        
    
    subsampled = []
    for token_id in corpus:
        f = freq[token_id]
        p_keep = math.sqrt(threshold / f)
        seed = random.random()
        if seed < p_keep:
            subsampled.append(token_id)
        
    return subsampled

def generate_skipgram_pairs(subsampled: list[int], window_size: int):
    for i in range(len(subsampled)):
        actual_window = random.randint(1,window_size)
        left = max(0, i - actual_window)
        right = min(len(subsampled), i+actual_window+1)
        for j in range(left,right):
            if j==i:
                continue 
            yield (subsampled[i], subsampled[j])        
            
            
# manifesting the generator into a list for efficient batch processing. 

class SkipGramDataset(torch.utils.data.Dataset):
    def __init__(self, subsampled, window_size):
        self.pairs = list(generate_skipgram_pairs(subsampled, window_size))
        
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        center, context = self.pairs[idx]
        return torch.as_tensor(center), torch.as_tensor(context)
        