
import csv
import torch

def read_ticker(ticker):
  values = []
  with open('{}.csv'.format(ticker), newline='') as csvfile:
      spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
      for row in spamreader:          
          try:
            values.append(float(row[5]))
          except:
             pass

  return torch.Tensor(values)

if __name__ == "__main__":
  data = read_ticker("AAPL")
  print(data)