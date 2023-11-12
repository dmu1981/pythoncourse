import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import math 
import random
from tqdm import tqdm
import transformer
import sequenceset
import swipes as swp
import balancedaccuracy

writer = SummaryWriter()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device is {DEVICE}")

swipes = swp.Swipes("training-data_50_swipes.txt", max_lines=512) 
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
  
dataloader = torch.utils.data.DataLoader(swipes, batch_size=BATCH_SIZE, shuffle=True)      

DROPOUT = False

def token_to_char(token):
  for key in indices.keys():
    if indices[key] == token:
      return key
      
  return "<UNK>"

net = transformer.Transformer(
   seq_len=SEQ_LEN, 
   n_tokens=N_TOKENS, 
   pad_token=PAD_TOKEN, 
   emb_dim=EMB_DIM, 
   intermediate_dim=EMB_DIM*4, 
   n_layers=2, 
   n_heads=6, 
   dropout=False).to(DEVICE)

optim = torch.optim.AdamW(net.parameters(), lr=0.001, weight_decay=0.01)
criterion = torch.nn.CrossEntropyLoss(reduction="none")

try:
    chkpt = torch.load("swipes8.pt")
    net.load_state_dict(chkpt["model"])
    optim.load_state_dict(chkpt["optim"])
except:
    print("Could not load model, starting from scratch!")

bar = tqdm(range(1,1000000))
total_loss = 0
cnt = 0
nrm = 0

balancedAccuracy = balancedaccuracy.BalancedAccuracy([PAD_TOKEN, SOS_TOKEN])

dataiterator =  dataloader.__iter__()
for iter in bar:  
  try:
    inp, tgt, labels = dataiterator.__next__()
  except:
    dataiterator = dataloader.__iter__()
    inp, tgt, labels = dataiterator.__next__()    

  labels = labels.view(-1)

  optim.zero_grad()
  x = net(inp, tgt)
  loss = criterion(x, labels)
  w = torch.Tensor([token_weights[token.item()] for token in labels]).to(DEVICE)
  loss = torch.sum(loss * w)

  loss.backward()

  balancedAccuracy.update(torch.argmax(x, dim=1).tolist(), labels.tolist())

  total_loss += loss.item()
  cnt += tgt.shape[0]
  nrm += torch.nn.utils.clip_grad_norm_(net.parameters(), 1000)
  optim.step()

  if iter % 100 == 0:
    bacc_dct, bacc, acc = balancedAccuracy.get()
    bacc_dct = { token_to_char(token): bacc_dct[token] for token in bacc_dct.keys() }
    balancedAccuracy.reset()

    bar.set_description(f"loss={total_loss / cnt * 1000.0:.3f}, acc={acc * 100:.3f}%, bacc={100*bacc:.3f}%, norm={tgt.shape[0] * nrm / cnt:.3f}")

    writer.add_scalar("loss", total_loss / cnt * 1000.0, iter)
    writer.add_scalar("accuracy", acc * 100, iter)
    writer.add_scalar("gradient_norm", tgt.shape[0] * nrm / cnt, iter)
    writer.add_scalar("balanced_accuracy", bacc * 100, iter)

    writer.add_scalars("token_accuracy", bacc_dct, iter)      

    total_loss = 0
    cnt, nrm = 0, 0

    try:
      chkpt = torch.save({
        "model": net.state_dict(),
        "optim": optim.state_dict(),
      }, "swipes8.pt")
    except:
      print("Could not save model!")
