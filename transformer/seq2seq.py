import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import math 
import random
from tqdm import tqdm
import transformer
import sequenceset
import swipes as swp

writer = SummaryWriter()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device is {DEVICE}")

swipes = swp.Swipes("training-data_50_swipes.txt", max_lines=None) 
indices = swipes.indices
token_weights = swipes.token_weights

PAD_TOKEN = swipes.pad_index
SOS_TOKEN = swipes.sos_index
EOS_TOKEN = swipes.eos_index
BATCH_SIZE = 256
EMB_DIM = 60
SEQ_LEN = swp.MAX_SEQ_LEN
N_TOKENS = swipes.next_index

print(swipes.indices)
swipes = sequenceset.Seq2SeqDataset(swipes, swipes.sos_index, swipes.eos_index, swipes.pad_index, SEQ_LEN, 1)
  
dataloader = torch.utils.data.DataLoader(swipes, batch_size=BATCH_SIZE*5, shuffle=True)      





DROPOUT = False
   
# class RandomSequences(torch.utils.data.Dataset):
#     def __init__(self, seq_len):
#         self.seq_len = seq_len
#         pass
    
#     def __len__(self):
#         return BATCH_SIZE
    
#     def __getitem__(self, idx):
#         len = round(random.uniform(4, self.seq_len//2))*2

#         seq1 = torch.trunc(EOS_TOKEN + 1 + torch.rand(len) * (N_TOKENS - EOS_TOKEN - 1)).type(torch.long).to(DEVICE)

#         seq2 = torch.zeros(seq1.shape[0] // 2).to(DEVICE)
#         for idx in range(0, seq1.shape[0]-1, 2):
#           seq2[idx//2] = (seq1[idx] + seq1[idx+1]) % (N_TOKENS - 3) + 3

#         seq2 = torch.flip(seq1, dims=[0])
#         #print("1:", seq1)
#         #print("2:", seq2)

#         return seq1, seq2

def token_to_char(token):
  for key in indices.keys():
    if indices[key] == token:
      return key
      
  return "<UNK>"

def oversample_based_on_bacc(dataiterator, dataloader, bacc, BS):
  try:
    inp, tgt, labels = dataiterator.__next__()
  except:
    dataiterator = dataloader.__iter__()
    inp, tgt, labels = dataiterator.__next__()

  if bacc is not None:
    w = torch.Tensor([1.0-bacc[token.item()]/100.0 for token in labels])
    w /= torch.sum(w)

    idx = w.multinomial(num_samples=BS, replacement=True)
    #print(idx)
    #_, idx = torch.sort(w, descending=False)
    #idx = idx[:BS]

    return inp[idx], tgt[idx], labels[idx], dataiterator
  
  return inp[:BS], tgt[:BS], labels[:BS], dataiterator

net = transformer.Transformer(
   seq_len=SEQ_LEN, 
   n_tokens=N_TOKENS, 
   pad_token=PAD_TOKEN, 
   emb_dim=EMB_DIM, 
   intermediate_dim=EMB_DIM*4, 
   n_layers=3, 
   n_heads=6, 
   dropout=False).to(DEVICE)

# dataset = sequenceset.Seq2SeqDataset(RandomSequences(SEQ_LEN), SOS_TOKEN, EOS_TOKEN, PAD_TOKEN, SEQ_LEN, 1)
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE)

optim = torch.optim.AdamW(net.parameters(), lr=0.001, weight_decay=0.01)
criterion = torch.nn.CrossEntropyLoss(reduction="none")

warmup = transformer.Warmup(optim=optim, target_lr=0.001, warmup_steps=200, min_lr=0.0001, cooldown_steps=100000)

try:
    chkpt = torch.load("swipes6.pt")
    net.load_state_dict(chkpt["model"])
    optim.load_state_dict(chkpt["optim"])
    warmup.load_state_dict(chkpt["scheduler"])
except:
    print("Could not load model, starting from scratch!")

bar = tqdm(range(1,1000000))
total_loss = 0
cnt = 0
acc = 0
nrm = 0

token_cnt = {}
token_acc = {}
for token in indices.values():
   if token == PAD_TOKEN or token == SOS_TOKEN:
       continue
   
   token_acc[token] = 0
   token_cnt[token] = 0

dl =  dataloader.__iter__()
bacc_dct = None
cnt2 = 0
for iter in bar:  
  inp, tgt, labels, dl = oversample_based_on_bacc(dl, dataloader, bacc_dct, BATCH_SIZE)
  # try:
  #   inp, tgt, labels = dl.__next__()
  # except:
  #   dl = dataloader.__iter__()
  #   inp, tgt, labels = dl.__next__()

  labels = labels.view(-1)

  optim.zero_grad()
  x = net(inp, tgt)
  loss = criterion(x, labels)
  w = torch.Tensor([token_weights[token.item()] for token in labels]).to(DEVICE)
  loss = torch.sum(loss * w)

  loss.backward()

  for token in indices.values():
     if token == PAD_TOKEN or token == SOS_TOKEN:
       continue
     
     idx = (labels == token)
     token_acc[token] += torch.sum(torch.argmax(x[idx], dim=1) == token).item()
     token_cnt[token] += torch.sum(idx).item()

  
  acc += torch.sum(torch.argmax(x, dim=1) == labels)

  total_loss += loss.item()
  cnt += tgt.shape[0]
  cnt2 += 1
  nrm += torch.nn.utils.clip_grad_norm_(net.parameters(), 1000)
  optim.step()

  warmup.step()

  if iter % 50 == 25 and cnt2 > 100:
    bacc_dct = { token: 100.0 * token_acc[token] / (token_cnt[token]+0.001) for token in token_acc.keys() }

  if iter % 50 == 0:
    acc_dct = { token_to_char(token): 100.0 * token_acc[token] / (token_cnt[token]+0.001) for token in token_acc.keys() }
    
    bacc = 0
    for token in acc_dct.keys():
      bacc += acc_dct[token] 
    bacc /= len(indices.values())    

    bar.set_description(f"loss={total_loss / cnt * 1000.0:.3f}, acc={acc / cnt * 100:.3f}%, bacc={bacc:.3f}%, norm={tgt.shape[0] * nrm / cnt:.3f}")

    writer.add_scalar("loss", total_loss / cnt * 1000.0, iter)
    writer.add_scalar("accuracy", acc / cnt * 100, iter)
    writer.add_scalar("gradient_norm", tgt.shape[0] * nrm / cnt, iter)
    writer.add_scalar("balanced_accuracy", bacc, iter)

    if iter % 500 == 0:    
      writer.add_scalars("token_accuracy", acc_dct, iter)      

      cnt2 = 0
      token_cnt = {}
      token_acc = {}
      for token in indices.values():
        if token == PAD_TOKEN or token == SOS_TOKEN:
          continue

        token_acc[token] = 0
        token_cnt[token] = 0

    total_loss = 0
    cnt = 0
    acc = 0
    nrm = 0
    
    


  if iter % 500 == 0:
    try:
      chkpt = torch.save({
        "model": net.state_dict(),
        "optim": optim.state_dict(),
        "scheduler": warmup.state_dict()
      }, "swipes6.pt")
    except:
      print("Could not save model!")
