import numpy as np

import torch
import torch.nn as nn

from configs import cfg

def get_embedder(n_vocab, dim):
    embedder = torch.nn.Embedding(num_embeddings=n_vocab, embedding_dim=dim)
    return embedder, dim