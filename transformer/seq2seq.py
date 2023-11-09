import torch
from torch import nn
import math 
import random
from tqdm import tqdm
import transformer
import sequenceset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device is {DEVICE}")

BATCH_SIZE = 500
EMB_DIM = 64
SEQ_LEN = 32
N_TOKENS = 32
PAD_TOKEN = 0
SOS_TOKEN = 1
EOS_TOKEN = 2

DROPOUT = False
   
class RandomSequences(torch.utils.data.Dataset):
    def __init__(self, seq_len):
        self.seq_len = seq_len
        pass
    
    def __len__(self):
        return BATCH_SIZE
    
    def __getitem__(self, idx):
        len = round(random.uniform(4, self.seq_len//2))*2

        seq1 = torch.trunc(EOS_TOKEN + 1 + torch.rand(len) * (N_TOKENS - EOS_TOKEN - 1)).type(torch.long).to(DEVICE)

        seq2 = torch.zeros(seq1.shape[0] // 2).to(DEVICE)
        for idx in range(0, seq1.shape[0]-1, 2):
          seq2[idx//2] = (seq1[idx] + seq1[idx+1]) % (N_TOKENS - 3) + 3

        seq2 = torch.flip(seq1, dims=[0])
        #print("1:", seq1)
        #print("2:", seq2)

        return seq1, seq2


net = transformer.Transformer(
   seq_len=SEQ_LEN, 
   n_tokens=N_TOKENS, 
   pad_token=PAD_TOKEN, 
   emb_dim=EMB_DIM, 
   intermediate_dim=EMB_DIM*4, 
   n_layers=2, 
   n_heads=4, 
   dropout=False).to(DEVICE)

dataset = sequenceset.Seq2SeqDataset(RandomSequences(SEQ_LEN), SOS_TOKEN, EOS_TOKEN, PAD_TOKEN, SEQ_LEN, 1)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE)

optim = torch.optim.Adam(net.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

warmup = transformer.Warmup(optim=optim, target_lr=0.01, warmup_steps=10000, min_lr=0.000001, cooldown_steps=100000)

try:
    chkpt = torch.load("seq2seq.pt")
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
      }, "seq2seq.pt")
    except:
      print("Could not save model!")
