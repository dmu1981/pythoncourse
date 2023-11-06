import torch
from torch import nn
import math 
import random
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device is {DEVICE}")

BATCH_SIZE = 256
EMB_DIM = 256
SEQ_LEN = 80

class SelfAttention(nn.Module):
    def __init__(self, emb_dim, n_heads = 8):
        super().__init__()

        self.n_heads = n_heads
        self.att_dim = emb_dim // n_heads
        self.n_dim_norm = math.sqrt(self.att_dim)

        self.Q = nn.Linear(emb_dim, emb_dim)
        self.K = nn.Linear(emb_dim, emb_dim)
        self.V = nn.Linear(emb_dim, emb_dim)

        # self.Q = [nn.Linear(self.att_dim, self.att_dim).to(DEVICE) for _ in range(n_heads)]
        # self.K = [nn.Linear(self.att_dim, self.att_dim).to(DEVICE) for _ in range(n_heads)]
        # self.V = [nn.Linear(self.att_dim, self.att_dim).to(DEVICE) for _ in range(n_heads)]

        pass

    def forward(self, x):
        res = torch.empty_like(x)

        Q = self.Q(x)
        K = self.K(x)
        V = self.V(x)

        for indx in range(self.n_heads):
            q_slice = Q[:,:,(indx*self.att_dim):((indx+1)*self.att_dim)]
            k_slice = torch.transpose(K[:,:,(indx*self.att_dim):((indx+1)*self.att_dim)], dim0=1,dim1=2)
            v_slice = V[:,:,(indx*self.att_dim):((indx+1)*self.att_dim)]
            
            # res = torch.softmax((q_slice @ k_slice) / self.n_dim_norm, dim=-1) @ v_slice
            # print(res.shape)
            # print(res)
            # exit()

            res[:,:,(indx*self.att_dim):((indx+1)*self.att_dim)] = torch.matmul(torch.softmax((q_slice @ k_slice) / self.n_dim_norm, dim=2), v_slice)

            # q = self.Q[indx](x_slice)
            # k = torch.transpose(self.K[indx](x_slice), dim0=1, dim1=2)
            # v = self.V[indx](x_slice)

            # res[:,:,(indx*self.att_dim):((indx+1)*self.att_dim)] = torch.matmul(torch.softmax((q @ k) / self.n_dim_norm, dim=2), v)
            
        return res
    
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

        self.selfAttention = SelfAttention(emb_dim=emb_dim, n_heads=n_heads).to(DEVICE)
        self.mlp = MLP(emb_dim=emb_dim, intermediate_dim=intermediate_dim).to(DEVICE)        

        self.layerNorm1 = nn.LayerNorm(emb_dim).to(DEVICE)
        self.layerNorm2 = nn.LayerNorm(emb_dim).to(DEVICE)

    def forward(self, x):
        x = self.layerNorm1(x + self.selfAttention(x))
        x = self.layerNorm2(x + self.mlp(x))
        return x
    
    
class Encoder(nn.Module):
    def __init__(self, n_tokens, emb_dim, intermediate_dim, n_layers=6, n_heads=8):
        super().__init__()

        self.emb_dim = emb_dim
        self.embedding = nn.Embedding(num_embeddings=n_tokens, embedding_dim=emb_dim)
        self.pos_encoding = nn.Embedding(num_embeddings=SEQ_LEN, embedding_dim=emb_dim)
        self.pos_indices = torch.Tensor(list(range(SEQ_LEN))).type(torch.long).to(DEVICE)
        
        self.encoders = [
            EncoderLayer(emb_dim=emb_dim, intermediate_dim=intermediate_dim, n_heads=n_heads) 
            for _ in range(n_layers)
            ]

    def forward(self, x):
        # i = self.pos_encoding(torch.Tensor([0]).type(torch.long).to(DEVICE))
        # print(i)
        # Token encoding
        x = self.embedding(x)
        
        # Positional encoding
        pos = self.pos_encoding(self.pos_indices[:x.shape[1]])# / math.sqrt(self.emb_dim)
        x += pos 


        # Encoder
        for encoder in self.encoders:
            x = encoder(x)

        return x

class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = Encoder(n_tokens=20, emb_dim=64, intermediate_dim=256, n_layers=6, n_heads=8)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(64, 2)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.encoder(x)[:,0,:]
        #print(x.shape)
        #x = self.flatten(x)
        return self.fc(x)
    
class SymmetricSequences(torch.utils.data.Dataset):
    def __init__(self, seq_len):
        self.seq_len = seq_len
        pass
    
    def __len__(self):
        return BATCH_SIZE
    
    def __getitem__(self, idx):
        seq = torch.trunc(torch.rand(self.seq_len) * 20).type(torch.long).to(DEVICE)
        if random.uniform(0, 1) < .5:
          seq[self.seq_len//2:] = torch.flip(seq[:-self.seq_len//2], dims=[0])
          return seq, torch.Tensor([1]).type(torch.long).to(DEVICE)
        else:
          return seq, torch.Tensor([0]).type(torch.long).to(DEVICE)

class Warmup():
    def __init__(self, optim, target_lr = 0.01, min_lr = 0.0001, warmup_steps = 1000, cooldown_steps = 10000):
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


dataset = SymmetricSequences(40)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE)

# batch, labels = dataloader.__iter__().__next__()
# x = net(batch)
# labels=labels.view(-1)
# loss = criterion(x, labels)

# print(batch, labels)
# print(x, loss)
#exit()

warmup = Warmup(optim=optim, target_lr=0.01, warmup_steps=2000, min_lr=0.0001, cooldown_steps=100000)

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
  batch, labels = dataloader.__iter__().__next__()
  labels = labels.view(-1)

  optim.zero_grad()
  # print(batch, labels)
  # exit()
  x = net(batch)
  loss = criterion(x, labels)

  loss.backward()
  acc += torch.sum(torch.argmax(x, dim=1) == labels)
  total_loss += loss.item()
  cnt += batch.shape[0]
  optim.step()
  warmup.step()

  if iter % 50 == 0:
    # print(batch)
    # print(labels)
    #print(x)
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

