import torch
from torch.utils.data import DataLoader
from device import DEVICE
from transform import validation_transform
from dataset import CatsDogsDataSet, TEST_SET_FOLDER, DataSetMode
import json
import csv
from network import Network
from trainloop import Trainer
from cache import DatasetCache
from checkpoints import CheckpointTrainer
from tqdm import tqdm


if __name__ == "__main__":
    dataset = CatsDogsDataSet(
        TEST_SET_FOLDER, max_samples_per_class=None, mode=DataSetMode.Test, transform=validation_transform) 
    
    dataloader = DataLoader(dataset, batch_size=200, shuffle=False)
    
    net = Network().to(DEVICE)
    net.eval()
    chkpt = torch.load("model.pt")
    net.load_state_dict(chkpt["net_state_dict"])

    predictions = torch.zeros(len(dataset)).to(DEVICE)

    with torch.no_grad():
      for batch, labels in tqdm(dataloader):
          labels = labels.reshape(-1)
          prediction = torch.softmax(net(batch), dim=1)[:,1]
          predictions[labels] = prediction

    with open("submission.csv", "w", newline='') as f:
      writer = csv.writer(f)
      writer.writerow(["id", "label"])
      for index, prediction in enumerate(predictions):
         writer.writerow([index+1, prediction.item()])
         
        
    

        