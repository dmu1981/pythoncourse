import torch
from torch import nn
import math 
import random
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PositionalEncoding(nn.Module):
    def __init__(self, seq_len, emb_dim):
        super().__init__()

        self.seq_len = seq_len
        self.emb_dim = emb_dim

        self.encodings = torch.zeros(seq_len, emb_dim)
        for dim_index in range(0, emb_dim, 2):
            freq = 10000 ** (dim_index / emb_dim)
            
            for position_index in range(seq_len):
                self.encodings[position_index][dim_index+0] = math.sin(position_index / freq)
                self.encodings[position_index][dim_index+1] = math.cos(position_index / freq)

        self.encodings = self.encodings.to(DEVICE) * math.sqrt(emb_dim)
        self.register_buffer("pe", self.encodings)

    def forward(self, x):
        return x + self.encodings[:x.shape[1],:]


class SelfAttention(nn.Module):
    def __init__(self, emb_dim, n_heads = 8, dropout = True):
        super().__init__()

        self.n_heads = n_heads
        self.att_dim = emb_dim // n_heads
        self.n_dim_norm = math.sqrt(self.att_dim)

        self.Q = nn.Linear(emb_dim, emb_dim)
        self.K = nn.Linear(emb_dim, emb_dim)
        self.V = nn.Linear(emb_dim, emb_dim)
        self.out = nn.Linear(emb_dim, emb_dim)
        self.dropout = nn.Identity() # nn.Dropout(0.1) if dropout else nn.Identity()

    def forward(self, xv, xk, xq, mask=None):
        
        res = torch.empty_like(xq)

        Q = self.Q(xq)
        K = self.K(xk)
        V = self.V(xv)

        for indx in range(self.n_heads):
            q_slice = Q[:,:,(indx*self.att_dim):((indx+1)*self.att_dim)]
            k_slice = torch.transpose(K[:,:,(indx*self.att_dim):((indx+1)*self.att_dim)], dim0=1,dim1=2)
            v_slice = V[:,:,(indx*self.att_dim):((indx+1)*self.att_dim)]

            scores = (q_slice @ k_slice) / self.n_dim_norm
            if mask is not None:
              scores = scores.masked_fill(mask == 0, -1e9)

            res[:,:,(indx*self.att_dim):((indx+1)*self.att_dim)] = self.dropout(
                torch.softmax(scores, dim=2)
                ) @ v_slice
            
        return self.out(res)
    
class MLP(nn.Module):
    def __init__(self, emb_dim, intermediate_dim, dropout=True):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, intermediate_dim),
            nn.ReLU(),
            nn.Dropout(0.1) if dropout else nn.Identity(),
            nn.Linear(intermediate_dim, emb_dim))
        
    def forward(self, x):
        return self.mlp(x)
    
class EncoderLayer(nn.Module):
    def __init__(self, emb_dim, intermediate_dim, n_heads=8):
        super().__init__()

        self.selfAttention = SelfAttention(emb_dim=emb_dim, n_heads=n_heads).to(DEVICE)
        self.mlp = MLP(emb_dim=emb_dim, intermediate_dim=intermediate_dim).to(DEVICE)        

        self.layerNorm1 = nn.LayerNorm(emb_dim).to(DEVICE)
        self.layerNorm2 = nn.LayerNorm(emb_dim).to(DEVICE)

    def forward(self, x, mask=None):
        x = self.layerNorm1(x + self.selfAttention(x, x, x, mask))
        x = self.layerNorm2(x + self.mlp(x))
        return x   

class DecoderLayer(nn.Module):
    def __init__(self, emb_dim, intermediate_dim, n_heads=8):
        super().__init__()

        self.selfAttention = SelfAttention(emb_dim=emb_dim, n_heads=n_heads).to(DEVICE)
        self.encoderAttention = SelfAttention(emb_dim=emb_dim, n_heads=n_heads).to(DEVICE)
        self.mlp = MLP(emb_dim=emb_dim, intermediate_dim=intermediate_dim).to(DEVICE)        

        self.layerNorm1 = nn.LayerNorm(emb_dim).to(DEVICE)
        self.layerNorm2 = nn.LayerNorm(emb_dim).to(DEVICE)
        self.layerNorm3 = nn.LayerNorm(emb_dim).to(DEVICE)

    def forward(self, x, encoded, mask=None):
        x = self.layerNorm1(x + self.selfAttention(x, x, x, mask))
        x = self.layerNorm2(x + self.selfAttention(encoded, encoded, x))
        x = self.layerNorm3(x + self.mlp(x))
        return x       
    
class Encoder(nn.Module):
    def __init__(self, seq_len, n_tokens, emb_dim, intermediate_dim, n_layers=6, n_heads=8, dropout=True):
        super().__init__()

        self.seq_len = seq_len
        self.emb_dim = emb_dim
        self.embedding = nn.Embedding(num_embeddings=n_tokens, embedding_dim=emb_dim)        
        self.pos_encoding = PositionalEncoding(seq_len, emb_dim=emb_dim)

        self.dropout = nn.Dropout(0.2) if dropout else nn.Identity()
        
        self.encoders = nn.ModuleList([
            EncoderLayer(emb_dim=emb_dim, intermediate_dim=intermediate_dim, n_heads=n_heads) 
            for _ in range(n_layers)
            ])

    def forward(self, x, src_mask):
        # Embeddings and positional encoding
        x = self.dropout(self.embedding(x))
        x = self.pos_encoding(x)       

        # Encoder
        for encoder in self.encoders:
            x = encoder(x, src_mask)

        return x
    
class Decoder(nn.Module):
    def __init__(self, seq_len, n_tokens, emb_dim, intermediate_dim, n_layers=6, n_heads=8, dropout=True):
        super().__init__()

        self.seq_len = seq_len

        self.emb_dim = emb_dim
        self.embedding = nn.Embedding(num_embeddings=n_tokens, embedding_dim=emb_dim)

        self.pos_encoding = PositionalEncoding(self.seq_len, emb_dim=emb_dim)

        self.dropout = nn.Dropout(0.2) if dropout else nn.Identity()
        
        self.decoders = nn.ModuleList([
            DecoderLayer(emb_dim=emb_dim, intermediate_dim=intermediate_dim, n_heads=n_heads) 
            for _ in range(n_layers)
            ])

    def forward(self, x, encoded):
        seq_len = x.shape[1]
        mask = 1.0 - torch.triu(torch.ones((seq_len, seq_len))).to(DEVICE)

        # Embeddings and positional encoding
        x = self.dropout(self.embedding(x))
        x = self.pos_encoding(x)

        # Encoder
        for decoder in self.decoders:
            x = decoder(x, encoded, mask)
 
        return x    

class Transformer(nn.Module):
    def __init__(self, seq_len, n_tokens, pad_token, emb_dim=256, intermediate_dim=1024, n_layers=6, n_heads=8, dropout=True):
        super().__init__()

        self.seq_len = seq_len
        self.pad_token = pad_token

        self.encoder = Encoder(seq_len=seq_len, n_tokens=n_tokens, emb_dim=emb_dim, intermediate_dim=intermediate_dim, n_layers=n_layers, n_heads=n_heads, dropout=dropout)
        self.decoder = Decoder(seq_len=seq_len, n_tokens=n_tokens, emb_dim=emb_dim, intermediate_dim=intermediate_dim, n_layers=n_layers, n_heads=n_heads, dropout=dropout)

    def forward(self, source_sequence, target_sequence):
        BS = source_sequence.shape[0]
        src_mask = torch.empty(BS, self.seq_len, self.seq_len).to(DEVICE)
        mask = 1.0 - 1.0 * (source_sequence == self.pad_token)
        for idx in range(source_sequence.shape[0]):
            src_mask[idx,:,:] = torch.outer(mask[idx], mask[idx])

        encoded = self.encoder(source_sequence, src_mask)
        return self.decoder(target_sequence, encoded)
    
        

class Warmup():
    def __init__(self, optim, target_lr = 0.01, min_lr = 0.0001, warmup_steps = 200, cooldown_steps = 5000):
        self.optim = optim
        self.target_lr = target_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.cooldown_steps = cooldown_steps
        self.steps = 0

    def state_dict(self):
        return { "steps": self.steps }
    
    def load_state_dict(self, dct):
        self.steps = dct["steps"]

    def step(self):
        self.steps += 1
        if self.steps < self.warmup_steps:
          ratio = self.steps / self.warmup_steps          
          lr = self.min_lr + ratio * (self.target_lr - self.min_lr)
        elif self.steps < self.cooldown_steps:
          ratio = 1.0 - (self.steps - self.warmup_steps) / (self.cooldown_steps - self.warmup_steps)
          lr = self.min_lr + ratio * (self.target_lr - self.min_lr)
        else:
          lr = self.min_lr

        for param_group in self.optim.param_groups:
            param_group['lr'] = lr





