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

class DimDropout(nn.Module):
    def __init__(self, dim, p=0.5):
        super().__init__()

        self.dim = dim
        self.p=p

    def forward(self, x):
        if self.training:
          shape = list(x.shape)
          del shape[self.dim]
          shape=torch.Size(shape)

          res = (torch.rand(size=shape, device=x.device) > self.p).unsqueeze(self.dim)
          shape = [1 for _ in range(len(res.shape))]
          shape[self.dim] = x.shape[self.dim]
          res = res.repeat(torch.Size(shape))

          return x * res
        else:
          return x

class SelfAttention(nn.Module):
    def __init__(self, emb_dim, n_heads = 8, dropout = True):
        super().__init__()

        self.n_heads = n_heads
        self.att_dim = emb_dim // n_heads
        self.n_dim_norm = math.sqrt(self.att_dim)

        self.dropout = DimDropout(1, 0.5) if dropout else nn.Identity()

        self.Q = nn.Linear(emb_dim, emb_dim)
        self.K = nn.Linear(emb_dim, emb_dim)
        self.V = nn.Linear(emb_dim, emb_dim)
        self.out = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.GELU()
        )

    def forward(self, xv, xk, xq, mask=None):
        
        res = torch.empty_like(xq)

        Q = self.Q(xq)
        K = self.K(xk)
        V = self.V(xv)

        for indx in range(self.n_heads):
            q_slice = Q[:,:,(indx*self.att_dim):((indx+1)*self.att_dim)]
            k_slice = torch.transpose(K[:,:,(indx*self.att_dim):((indx+1)*self.att_dim)], dim0=1,dim1=2)
            v_slice = V[:,:,(indx*self.att_dim):((indx+1)*self.att_dim)]

            k_slice = self.dropout(k_slice)

            scores = (q_slice @ k_slice) / self.n_dim_norm
            if mask is not None:
              scores = scores.masked_fill(mask == 0, -1e9)

            res[:,:,(indx*self.att_dim):((indx+1)*self.att_dim)] = torch.softmax(scores, dim=2) @ v_slice
            
        return self.out(res)
    
class MLP(nn.Module):
    def __init__(self, emb_dim, intermediate_dim, dropout=True):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, intermediate_dim),
            nn.GELU(),
            nn.Dropout(0.5) if dropout else nn.Identity(),
            nn.Linear(intermediate_dim, emb_dim),
            nn.GELU()
            )
        
    def forward(self, x):
        return self.mlp(x)

class ResidualBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.Tensor([0.01]), requires_grad=True)

    def forward(self, xIn, xOut):
        return xIn + self.alpha * xOut

class EncoderLayer(nn.Module):
    def __init__(self, emb_dim, intermediate_dim, n_heads=8):
        super().__init__()

        self.selfAttention = SelfAttention(emb_dim=emb_dim, n_heads=n_heads).to(DEVICE)
        self.mlp = MLP(emb_dim=emb_dim, intermediate_dim=intermediate_dim).to(DEVICE)        

        self.residual1 = ResidualBlock()
        self.residual2 = ResidualBlock()

    def forward(self, x, mask=None):
        x = self.residual1(x, self.selfAttention(x, x, x, mask))
        x = self.residual2(x, self.mlp(x))

        return x   

class DecoderLayer(nn.Module):
    def __init__(self, emb_dim, intermediate_dim, n_heads=8):
        super().__init__()

        self.selfAttention = SelfAttention(emb_dim=emb_dim, n_heads=n_heads).to(DEVICE)
        self.encoderAttention = SelfAttention(emb_dim=emb_dim, n_heads=n_heads).to(DEVICE)
        self.mlp = MLP(emb_dim=emb_dim, intermediate_dim=intermediate_dim).to(DEVICE)        

        self.residual1 = ResidualBlock()
        self.residual2 = ResidualBlock()
        self.residual3 = ResidualBlock()

    def forward(self, x, encoded, mask=None):
        x = self.residual1(x, self.selfAttention(x, x, x, mask))
        x = self.residual2(x, self.selfAttention(encoded, encoded, x, mask))
        x = self.residual3(x, self.mlp(x))

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
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, intermediate_dim),
            nn.GELU(),
            nn.Linear(intermediate_dim, n_tokens)
            )

    def forward(self, source_sequence, target_sequence):
        encoded = self.encode(source_sequence)
        return self.decode(target_sequence, encoded)

    def decode(self, target_sequence, encoded):
        decoded = self.decoder(target_sequence, encoded)

        return self.mlp(decoded[:,0,:])
    
    def encode(self, source_sequence):
        BS = source_sequence.shape[0]
        src_mask = torch.empty(BS, self.seq_len, self.seq_len).to(DEVICE)
        mask = 1.0 - 1.0 * (source_sequence == self.pad_token)
        for idx in range(source_sequence.shape[0]):
            src_mask[idx,:,:] = torch.outer(mask[idx], mask[idx])

        return self.encoder(source_sequence, src_mask)
        
    
    def seq2seq(self, source_sequence, sos_token, eos_token):
        target_sequence = torch.ones(self.seq_len).to(source_sequence.device).to(torch.long).view(1,-1) * self.pad_token
        target_sequence[0] = sos_token

        encoded = self.encode(source_sequence.view(1,-1))
        idx = 1
        while idx < self.seq_len:
          res = torch.argmax(self.decode(target_sequence, encoded))
          target_sequence[0,idx] = res
          idx += 1
          if res == eos_token:
              break

        return target_sequence[0]


        

