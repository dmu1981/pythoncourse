import torch
from torch import nn
import random
import math

class Seq2SeqDataset(torch.utils.data.Dataset):
  def __init__(self, baseSet, SOS_token, EOS_token, PAD_token, PAD_len, factor):
    super().__init__()

    self.baseSet = baseSet
    self.SOS = SOS_token
    self.EOS = EOS_token
    self.PAD = PAD_token
    self.PAD_len = PAD_len
    self.factor = factor

  def __len__(self):
    return len(self.baseSet) * self.factor
  
  def __getitem__(self, idx):
    # Normalize index
    idx = idx % len(self.baseSet)

    # Get the base sequence pair
    src, tgt, w = self.baseSet[idx]

    # Get the device we need to operate on
    dvc = tgt.device
    
    # Limit length so we can always append SOS and EOS
    tgt = tgt[:(self.PAD_len-2)]
    tgt = torch.concat((torch.Tensor([self.SOS]).to(dvc), tgt, torch.Tensor([self.EOS]).to(dvc)))

    # Pick a random substring
    cut = math.trunc(random.uniform(2, tgt.shape[0]))

    token_to_predict = tgt[cut]
    tgt = tgt[:cut]

    # Now pad the sequence
    src = torch.concat((src, torch.ones(self.PAD_len - src.shape[0]).to(dvc) * self.PAD)).type(torch.long)
    tgt = torch.concat((tgt, torch.ones(self.PAD_len - tgt.shape[0]).to(dvc) * self.PAD)).type(torch.long)

    return src, tgt, token_to_predict.type(torch.long).to(dvc), w
  