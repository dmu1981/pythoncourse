import torch
from torch.utils.data import DataLoader
from device import DEVICE
from transform import training_transform
from dataset import CatsDogsDataSet, TRAIN_SET_FOLDER
import numpy as np
from network import Network
from tqdm import tqdm

def epoch(dataloader, network, optim, loss_function, training):
  bar = tqdm(dataloader)
  
  if training:
     network.train()
  else:
     network.eval()

  total_loss = 0
  cnt = 0
  for batch, labels in bar:
     if training:
      optim.zero_grad()

     res = network(batch)
     loss = loss_function(res, labels.view(-1))
     total_loss += loss.item()
     cnt += batch.shape[0]

     bar.set_description(f"loss: {1000.0*total_loss / cnt:.3f}")

     if training:
        loss.backward()
        optim.step()

if __name__ == "__main__":
    dataset = CatsDogsDataSet(TRAIN_SET_FOLDER, max_samples_per_class=2000, transform=training_transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    net = Network().to(DEVICE)
    optim = torch.optim.Adam(net.parameters(), lr=0.001)
    loss = torch.nn.CrossEntropyLoss()
    
    epoch(dataloader, net, optim, loss, True)