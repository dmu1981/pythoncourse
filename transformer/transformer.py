import torch
from torch import nn
import math 
import random
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device is {DEVICE}")

BATCH_SIZE = 256
EMB_DIM = 256
SEQ_LEN = 12
N_TOKENS = 8
DROPOUT = 0

class SelfAttention(nn.Module):
    def __init__(self, emb_dim, n_heads = 8):
        super().__init__()

        self.n_heads = n_heads
        self.att_dim = emb_dim // n_heads
        self.n_dim_norm = math.sqrt(self.att_dim)

        self.Q = nn.Linear(emb_dim, emb_dim)
        self.K = nn.Linear(emb_dim, emb_dim)
        self.V = nn.Linear(emb_dim, emb_dim)
        self.out = nn.Linear(emb_dim, emb_dim)
        self.dropout = nn.Dropout(0.1 * DROPOUT)

    def forward(self, xq, xk, xv, mask=None):
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
    def __init__(self, emb_dim, intermediate_dim):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, intermediate_dim),
            nn.ReLU(),
            nn.Dropout(0.5 * DROPOUT),
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
    def __init__(self, n_tokens, emb_dim, intermediate_dim, n_layers=6, n_heads=8):
        super().__init__()

        self.emb_dim = emb_dim
        self.embedding = nn.Embedding(num_embeddings=n_tokens, embedding_dim=emb_dim)
        self.pos_encoding = nn.Embedding(num_embeddings=SEQ_LEN, embedding_dim=emb_dim)
        self.pos_indices = torch.Tensor(list(range(SEQ_LEN))).type(torch.long).to(DEVICE)
        self.dropout = nn.Dropout(0.2 * DROPOUT)
        
        self.encoders = [
            EncoderLayer(emb_dim=emb_dim, intermediate_dim=intermediate_dim, n_heads=n_heads) 
            for _ in range(n_layers)
            ]

    def forward(self, x):
        seq_len = x.shape[1]

        # Embeddings and positional encoding
        x = self.dropout(self.embedding(x)) + self.pos_encoding(self.pos_indices[:seq_len])

        # Encoder
        for encoder in self.encoders:
            x = encoder(x)

        return x
    
class Decoder(nn.Module):
    def __init__(self, n_tokens, emb_dim, intermediate_dim, n_layers=6, n_heads=8):
        super().__init__()

        self.emb_dim = emb_dim
        self.embedding = nn.Embedding(num_embeddings=n_tokens, embedding_dim=emb_dim)
        self.pos_encoding = nn.Embedding(num_embeddings=SEQ_LEN, embedding_dim=emb_dim)
        self.pos_indices = torch.Tensor(list(range(SEQ_LEN))).type(torch.long).to(DEVICE)
        self.dropout = nn.Dropout(0.2 * DROPOUT)
        
        self.decoders = [
            DecoderLayer(emb_dim=emb_dim, intermediate_dim=intermediate_dim, n_heads=n_heads) 
            for _ in range(n_layers)
            ]

    def forward(self, x, encoded):
        seq_len = x.shape[1]
        mask = 1.0 - torch.triu(torch.ones((seq_len, seq_len))).to(DEVICE)

        # Embeddings and positional encoding
        x = self.dropout(self.embedding(x)) + self.pos_encoding(self.pos_indices[:seq_len])

        # Encoder
        for decoder in self.decoders:
            x = decoder(x, encoded, mask)
 
        return x    

class Transformer(nn.Module):
    def __init__(self, n_tokens=N_TOKENS, emb_dim=EMB_DIM, intermediate_dim=EMB_DIM*4, n_layers=6, n_heads=8):
        super().__init__()

        self.encoder = Encoder(n_tokens=N_TOKENS, emb_dim=EMB_DIM, intermediate_dim=EMB_DIM*4, n_layers=6, n_heads=8)
        self.decoder = Decoder(n_tokens=N_TOKENS, emb_dim=EMB_DIM, intermediate_dim=EMB_DIM*4, n_layers=6, n_heads=8)

    def forward(self, source_sequence, target_sequence):
        encoded = self.encoder(source_sequence)
        return self.decoder(target_sequence, encoded)

class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.transformer = Transformer(n_tokens=N_TOKENS, emb_dim=EMB_DIM, intermediate_dim=EMB_DIM*4, n_layers=6, n_heads=8)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(EMB_DIM, 2)

    def forward(self, inp, tgt):
        x = self.transformer(inp, tgt)[:,0,:]
        return self.fc(x)
    
class SymmetricSequences(torch.utils.data.Dataset):
    def __init__(self, seq_len):
        self.seq_len = seq_len
        pass
    
    def __len__(self):
        return BATCH_SIZE
    
    def __getitem__(self, idx):        
        if random.uniform(0, 1) < .5:
            seq1 = torch.trunc(torch.rand(self.seq_len) * N_TOKENS).type(torch.long).to(DEVICE)
            seq2 = torch.flip(seq1, dims=[0])
            return seq1, seq2, torch.Tensor([1]).type(torch.long).to(DEVICE)
        else:
            seq1 = torch.trunc(torch.rand(self.seq_len) * N_TOKENS).type(torch.long).to(DEVICE)
            seq2 = torch.trunc(torch.rand(self.seq_len) * N_TOKENS).type(torch.long).to(DEVICE)
            return seq1, seq2, torch.Tensor([0]).type(torch.long).to(DEVICE)
        

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
          lr = self.target_lr * ratio
        elif self.steps < self.cooldown_steps:
          ratio = 1.0 - (self.steps - self.warmup_steps) / (self.cooldown_steps - self.warmup_steps)
          lr = self.min_lr + ratio * self.target_lr
        else:
          lr = self.min_lr

        for param_group in self.optim.param_groups:
            param_group['lr'] = lr

        

net = Net().to(DEVICE)

optim = torch.optim.Adam(net.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()
dataset = SymmetricSequences(SEQ_LEN)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE)
warmup = Warmup(optim=optim, target_lr=0.01, warmup_steps=6000, min_lr=0.0001, cooldown_steps=36000)

try:
    chkpt = torch.load("model.pt")
    net.load_state_dict(chkpt["model"])
    optim.load_state_dict(chkpt["optim"])
    warmup.load_state_dict(chkpt["scheduler"])
except:
    print("Could not load model, starting from scratch!")

bar = tqdm(range(1000000))
total_loss = 0
cnt = 0
acc = 0
for iter in bar:  
  inp, tgt, labels = dataloader.__iter__().__next__()
  labels = labels.view(-1)

  optim.zero_grad()
  x = net(inp, tgt)
  loss = criterion(x, labels)

  loss.backward()
  acc += torch.sum(torch.argmax(x, dim=1) == labels)
  total_loss += loss.item()
  cnt += tgt.shape[0]
  optim.step()
  warmup.step()

  if iter % 50 == 0:
    bar.set_description(f"loss={total_loss / cnt * 1000.0:.3f}, acc={acc / cnt * 100:.3f}%")
    total_loss = 0
    cnt = 0
    acc = 0

  if iter % 500 == 0:
    try:
      chkpt = torch.save({
        "model": net.state_dict(),
        "optim": optim.state_dict(),
        "scheduler": warmup.state_dict()
      }, "model.pt")
    except:
      print("Could not save model!")




exit()
        

input = torch.zeros(size=(BATCH_SIZE, SEQ_LEN), dtype=torch.long).to(DEVICE)
output = net(input)
print(output.shape)

