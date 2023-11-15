import csv
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import random

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device is {DEVICE}")

MAX_SEQ_LEN = 80

class Swipes(torch.utils.data.Dataset):
  def __len__(self):
    return self.input_swipes.shape[0]
  
  def __getitem__(self, index):
    return self.input_swipes[index], self.target_sequences[index], self.weights[index]
  
  def __init__(self, datafile, max_lines = 5000):
    self.indices = {}
    self.next_index = 0

    self.input_swipes = None
    self.target_sequences = None
    self.weights = None

    self.sos_index = self.char_to_index("<SOS>") # Star7t of SWIPE
    self.eos_index = self.char_to_index("<EOS>") # End of SWIPE

    lines = []
    with open(datafile, "rt") as f:
      reader = csv.reader(f, delimiter=",")
      for line in reader:
        lines.append(line)

    random.shuffle(lines)
    lines = lines[:max_lines]
    
    cnt = 0
    for line in tqdm(lines, desc="Reading SWIPES"):
        self.add(line[1], line[0], float(int(line[2])))
        cnt += 1

  def char_to_index(self, char):
    if char in self.indices:
      return self.indices[char]
    else:
      self.indices[char] = self.next_index
      self.next_index += 1
      return self.indices[char]

  def sequence_to_indices(self, sequence):
    seq = [self.char_to_index(x) for x in sequence]
    seq = seq[:(MAX_SEQ_LEN - 1)]
    while len(seq) < MAX_SEQ_LEN:
      seq.append(self.eos_index)

    return torch.tensor(seq).view(1,-1).to(DEVICE)

  def add(self, input, output, weight):
    inp = self.sequence_to_indices(input)
    if self.input_swipes is None:
      self.input_swipes = inp
    else:
      self.input_swipes = torch.cat((self.input_swipes, inp))

    out = self.sequence_to_indices(output)
    if self.target_sequences is None:
      self.target_sequences = out
    else:
      self.target_sequences = torch.cat((self.target_sequences, out))

    w = torch.torch.Tensor([weight]).view(1,-1).to(DEVICE)
    if self.weights is None:
      self.weights = w
    else:
      self.weights = torch.cat((self.weights, w))

swipes = Swipes("training-data_50_swipes.txt", max_lines=1000)
loader = DataLoader(swipes, batch_size=32, shuffle=True)      

input_sequence, output_sequence, weight = loader.__iter__().__next__()
print(input_sequence[0])
print(output_sequence[0])
#print(weight)
