from scipy.io import arff
import torch
from tqdm import tqdm
import read_ticker

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device is {DEVICE}")

class TickerDataSet(torch.utils.data.Dataset):
  def __init__(self, seqlen, lookahead=1):
    self.ticker = ["AAPL", "AMZN", "BRK-A", "GOOG", "JPM", "MSFT", "NFLX", "NVDA", "TSLA", "XOM"]
    self.ticker = { ticker: read_ticker.read_ticker(ticker) for ticker in self.ticker }
    
    self.data = None
    for tensor in self.ticker.values():
      if self.data is None:
        self.data = tensor.view(1,-1)
      else:
        self.data = torch.concat((self.data, tensor.view(1,-1)), dim=0)
    
    # Normalize data
    #self.data = (self.data - torch.mean(self.data, dim=1)) / torch.std(self.data, dim=1)

    print(self.data[0])

  def __len__(self):
    return 

class EEGDataSet(torch.utils.data.Dataset):
  def __init__(self, seqlen):
    self.data = arff.loadarff("eeg.arff")[0]
    print(self.data)

    self.eeg = None
    self.labels = None
    self.seqlen = seqlen

    for idx in tqdm(range(len(self.data))):
      eeg = torch.Tensor([self.data[idx][cnt] for cnt in range(14)]).to(DEVICE).view(1,-1)
      label = torch.Tensor([0 if self.data[idx][14] == b'0' else 1]).type(torch.long).to(DEVICE)

      if self.eeg is None:
        self.eeg = eeg
      else:
        self.eeg = torch.concat((self.eeg, eeg), dim=0)

      if self.labels is None:
        self.labels = label
      else:
        self.labels = torch.concat((self.labels, label), dim=0)

    # Normalize data
    self.eeg = (self.eeg - torch.mean(self.eeg, dim=0)) / torch.std(self.eeg, dim=0)

  def __len__(self):
    return len(self.data) - self.seqlen
  
  def __getitem__(self, idx):
    return self.eeg[idx:(idx+self.seqlen)], self.labels[idx+self.seqlen]

if __name__ == "__main__":
  dataset = TickerDataSet(seqlen=30, lookahead=1)
  print(dataset[0])