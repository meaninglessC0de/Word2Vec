import torch 
import torch.nn.functional as F
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

def n_nearest_neighbours(word, n, word2idx, idx2word, embeddings):
    index = word2idx[word]
    vector = embeddings[index]
    cosines = torch.mv(embeddings, vector) / (torch.norm(embeddings, dim = 1) * torch.norm(vector))
    indices = torch.argsort(cosines, descending=True)[1:n+1]
    return [idx2word[i.item()] for i in indices]




def plot_tsne(words, word2idx, embeddings):
    res = []
    for word in words:
        if word in word2idx:
            res.append(word)
    
    vectors = [embeddings[word2idx[word]] for word in res]
    vectors = torch.stack(vectors).cpu().detach().numpy()

    tsne = TSNE(2)
    coords = tsne.fit_transform(vectors)

    plt.figure(figsize=(12,8))
    for i, word in enumerate(res):
        x = coords[i,0]
        y = coords[i,1]
        plt.scatter(x,y)
        plt.annotate(word, (x,y))
        
    plt.show()
        