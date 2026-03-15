import numpy as np
import torch
import torch.nn as nn

class SkipGram(nn.Module):
    
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.input_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.output_embeddings = nn.Embedding(vocab_size, embed_dim)

        nn.init.uniform_(self.input_embeddings.weight, -0.5/embed_dim, 0.5/embed_dim)
        nn.init.uniform_(self.output_embeddings.weight, -0.5/embed_dim, 0.5/embed_dim)
    
    # sigmoid-based loss function that uses negative sampling 
    def forward(self,centre, context, negatives):
        center_vector = self.input_embeddings(centre)
        context_vector = self.output_embeddings(context)
        negative_vectors = self.output_embeddings(negatives)
        
        positive_score = (context_vector * center_vector).sum(dim=1)
        negative_score = torch.bmm(center_vector.unsqueeze(1), negative_vectors.transpose(1,2)).squeeze(1).sum(1)
        
        positive_loss = torch.log(torch.sigmoid(positive_score).clamp(min=1e-7))
        negative_loss = torch.log(torch.sigmoid(-negative_score).clamp(min=1e-7))
        loss = -(positive_loss.mean() + negative_loss.mean())
        return loss     