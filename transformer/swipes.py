
import csv
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import random
import sequenceset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device is {DEVICE}")

MAX_SEQ_LEN = 100

class Swipes(torch.utils.data.Dataset):
  def __len__(self):
    return len(self.input_swipes)
  
  def __getitem__(self, index):
    return self.input_swipes[index], self.target_sequences[index]#, self.weights[index]
  
  def __init__(self, datafile, max_lines = 5000):
    self.indices = {}
    self.count = {}
    self.next_index = 0

    self.input_swipes = []
    self.target_sequences = []
    self.sentence_weights = []


    self.max_seq_len = MAX_SEQ_LEN
    self.pad_index = self.char_to_index("PAD") # Padding Token
    self.sos_index = self.char_to_index("SOS") # Star7t of SWIPE
    self.eos_index = self.char_to_index("EOS") # End of SWIPE

    lines = []
    with open(datafile, "rt") as f:
      reader = csv.reader(f, delimiter=",")
      for line in reader:
        lines.append(line)

    random.shuffle(lines)
    if max_lines is not None:
      lines = lines[:max_lines]
    
    cnt = 0
    for line in tqdm(lines, desc="Reading SWIPES"):
        #for cnt in range(int(line[2])):
        self.add(line[1], line[0], float(int(line[2])))
        cnt += 1

    self.count["EOS"] = cnt
    total = 0
    minv = 99999999999
    for token, cnt in self.count.items():
      if cnt < minv and token != "PAD" and token != "SOS":
        minv = cnt
      total += cnt

    self.token_weights = { self.indices[token]: minv / self.count[token] for token in self.count }

  def char_to_index(self, char):
    if char in self.indices:
      self.count[char] += 1
      return self.indices[char]
    else:
      self.count[char] = 1
      self.indices[char] = self.next_index
      self.next_index += 1
      return self.indices[char]

  def sequence_to_indices(self, sequence):
    sequence = ''.join([c for c in sequence if c in ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]])
    seq = [self.char_to_index(x) for x in sequence]
    seq = seq[:(MAX_SEQ_LEN - 1)]
    #while len(seq) < MAX_SEQ_LEN:
      #seq.append(self.eos_index)

    return torch.tensor(seq).to(DEVICE)

  def add(self, input, output, weight):
    inp = self.sequence_to_indices(input)
    out = self.sequence_to_indices(output)
    self.input_swipes.append(inp)
    self.target_sequences.append(out)
    self.sentence_weights.append(weight)

if __name__ == "__main__":
  swipes = Swipes("training-data_50_swipes.txt", max_lines=10000) 
  print(swipes.indices)
  print(swipes.count)
  print(swipes.token_weights)
  swipes = sequenceset.Seq2SeqDataset(swipes, swipes.sos_index, swipes.eos_index, swipes.pad_index, 80, 1)
    
  loader = DataLoader(swipes, batch_size=32, shuffle=True)      

  input_sequence, output_sequence, target = loader.__iter__().__next__()
