import torch
from torch import nn
import math 
import random
from tqdm import tqdm
import transformer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device is {DEVICE}")

BATCH_SIZE = 500
EMB_DIM = 64
SEQ_LEN = 16
N_TOKENS = 16
PAD_TOKEN = N_TOKENS - 1

DROPOUT = False


class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.transformer = transformer.Transformer(seq_len=SEQ_LEN, n_tokens=N_TOKENS, pad_token=PAD_TOKEN, emb_dim=EMB_DIM, intermediate_dim=EMB_DIM*4, n_layers=2, n_heads=4, dropout=False)
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
        len = round(random.uniform(1, self.seq_len))

        LAST_TOKEN = N_TOKENS - 1
        if random.uniform(0, 1) < .5:
            seq1 = torch.trunc(torch.rand(len) * LAST_TOKEN).type(torch.long).to(DEVICE)
            seq2 = torch.flip(seq1, dims=[0])
            cls = 1
        else:
            seq1 = torch.trunc(torch.rand(len) * LAST_TOKEN).type(torch.long).to(DEVICE)
            seq2 = torch.trunc(torch.rand(len) * LAST_TOKEN).type(torch.long).to(DEVICE)
            cls = 0

        # Padding
        seq1 = torch.concat((seq1, torch.ones(self.seq_len - len).type(torch.long).to(DEVICE) * PAD_TOKEN))
        seq2 = torch.concat((seq2, torch.ones(self.seq_len - len).type(torch.long).to(DEVICE) * PAD_TOKEN))        

        return seq1, seq2, torch.Tensor([cls]).type(torch.long).to(DEVICE)



        

net = Net().to(DEVICE)


optim = torch.optim.Adam(net.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()
dataset = SymmetricSequences(SEQ_LEN)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE)
warmup = transformer.Warmup(optim=optim, target_lr=0.01, warmup_steps=10000, min_lr=0.000001, cooldown_steps=100000)

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
nrm = 0
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
  nrm += torch.nn.utils.clip_grad_norm_(net.parameters(), 0.2)
  optim.step()

  warmup.step()

  if iter % 50 == 0:
    bar.set_description(f"loss={total_loss / cnt * 1000.0:.3f}, acc={acc / cnt * 100:.3f}%, norm={tgt.shape[0] * nrm / cnt:.3f}")
    total_loss = 0
    cnt = 0
    acc = 0
    nrm = 0

  if iter % 50 == 0:
    try:
      chkpt = torch.save({
        "model": net.state_dict(),
        "optim": optim.state_dict(),
        "scheduler": warmup.state_dict()
      }, "model.pt")
    except:
      print("Could not save model!")
