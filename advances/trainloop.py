import torch
from torch.utils.data import DataLoader
from device import DEVICE
from transform import training_transform
from dataset import CatsDogsDataSet, TRAIN_SET_FOLDER
import numpy as np
from network import Network
from tqdm import tqdm

def epoch(dataloader, network, optim, loss_function, training, epoch=0):
  # We want a dedicated TQDM bar, so we can set the description after each step
  bar = tqdm(dataloader)
  
  # Tell the network whether we are training or evaluating (to disable DropOut)
  if training:
     network.train()
  else:
     network.eval()

  # This epoch starts
  total_loss = 0
  correct = 0
  cnt = 0

  # Iterate over the whole epoch
  for batch, labels in bar:
     # If we are training, zero out the gradients in the network
     if training:
      optim.zero_grad()

     # Do one forward pass
     res = network(batch)

     # Reshape labels for processing
     labels = labels.reshape(-1)

     # Calculcate the (BCE)-Loss
     loss = loss_function(res, labels)

     # Sum the total loss
     total_loss += loss.item()

     # Count how many correct predictions we have (for accuracy)
     correct += torch.sum(torch.argmax(res, dim=1) == labels).item()

     # Count total samples processed
     cnt += batch.shape[0]

     # Update bar description
     bar.set_description(f"ep: {epoch:.0f}, loss: {1000.0*total_loss / cnt:.3f}, acc: {100.0*correct/cnt:.2f}%")

     # If we are training, do backward pass 
     if training:
        # Calculcate backward gradients
        loss.backward()

        # Step the optimizer
        optim.step()

  return 1000.0 * total_loss / cnt, 100.0*correct/cnt

if __name__ == "__main__":
    dataset = CatsDogsDataSet(TRAIN_SET_FOLDER, max_samples_per_class=2000, transform=training_transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    net = Network().to(DEVICE)
    optim = torch.optim.Adam(net.parameters(), lr=0.001)
    loss = torch.nn.CrossEntropyLoss()
    
    epoch(dataloader, net, optim, loss, True)