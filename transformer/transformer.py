import torch
from torch import nn
import math 

BATCH_SIZE = 32
EMB_DIM = 256
SEQ_LEN = 80

class SelfAttention(nn.Module):
    def __init__(self, emb_dim, n_heads = 8):
        super().__init__()

        self.n_heads = n_heads
        self.att_dim = emb_dim // n_heads
        self.n_dim_norm = math.sqrt(self.att_dim)

        self.Q = [nn.Linear(self.att_dim, self.att_dim) for _ in range(n_heads)]
        self.K = [nn.Linear(self.att_dim, self.att_dim) for _ in range(n_heads)]
        self.V = [nn.Linear(self.att_dim, self.att_dim) for _ in range(n_heads)]

        pass

    def forward(self, x):
        res = torch.empty_like(x)

        for indx in range(self.n_heads):
            x_slice = x[:,:,(indx*self.att_dim):((indx+1)*self.att_dim)]
            q = self.Q[indx](x_slice)
            k = torch.transpose(self.K[indx](x_slice), dim0=1, dim1=2)
            v = self.V[indx](x_slice)

            res[:,:,(indx*self.att_dim):((indx+1)*self.att_dim)] = torch.softmax((q @ k) / self.n_dim_norm, dim=2) @ v
            
        return x
    
class MLP(nn.Module):
    def __init__(self, emb_dim, intermediate_dim):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, intermediate_dim),
            nn.ReLU(),
            nn.Linear(intermediate_dim, emb_dim))
        
    def forward(self, x):
        return self.mlp(x)
    
class EncoderLayer(nn.Module):
    def __init__(self, emb_dim, intermediate_dim, n_heads=8):
        super().__init__()

        self.selfAttention = SelfAttention(emb_dim=emb_dim, n_heads=n_heads)
        self.mlp = MLP(emb_dim=emb_dim, intermediate_dim=intermediate_dim)        

        self.layerNorm = nn.LayerNorm(emb_dim)

    def forward(self, x):
        x = self.layerNorm(x + self.selfAttention(x))
        x = self.layerNorm(x + self.mlp(x))
        return x
    
    
class Encoder(nn.Module):
    def __init__(self, n_tokens, emb_dim, intermediate_dim, n_layers=6, n_heads=8):
        super().__init__()

        self.embedding = nn.Embedding(num_embeddings=n_tokens, embedding_dim=emb_dim)

        self.encoders = [
            EncoderLayer(emb_dim=emb_dim, intermediate_dim=intermediate_dim, n_heads=n_heads) 
            for _ in range(n_layers)
            ]

    def forward(self, x):
        x = self.embedding(x)[:,:,0,:]

        print(x.shape)

        for encoder in self.encoders:
            x = encoder(x)

        return x
        

input = torch.zeros(size=(BATCH_SIZE, SEQ_LEN, 1), dtype=torch.long)

att = Encoder(n_tokens=64, emb_dim=EMB_DIM, intermediate_dim =4*EMB_DIM, n_layers=6, n_heads=8)
output = att(input)
print(output.shape)