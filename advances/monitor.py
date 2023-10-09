import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from device import DEVICE
from transform import training_transform, validation_transform
from dataset import CatsDogsDataSet, TRAIN_SET_FOLDER, DataSetMode
from cache import DatasetCache
import numpy as np
from network import Network
from checkpoints import CheckpointTrainer

class MonitoredTrainer(CheckpointTrainer):
    def __init__(self, network, loss_function, chkpt_path):
        super().__init__(network, loss_function, chkpt_path)

        self.writer = SummaryWriter()

    def logger(self, statistics):
        self.writer.add_scalar("train/loss", statistics["training"]["loss"], statistics["epoch"])
        self.writer.add_scalar("train/accuracy", statistics["training"]["accuracy"], statistics["epoch"])
        self.writer.add_scalar("validation/loss", statistics["validation"]["loss"], statistics["epoch"])
        self.writer.add_scalar("validation/accuracy", statistics["validation"]["accuracy"], statistics["epoch"])

if __name__ == "__main__":
    dataset = DatasetCache(
      CatsDogsDataSet(TRAIN_SET_FOLDER, max_samples_per_class=None, mode=DataSetMode.Final), 
      transform=training_transform)
    
    dataset_val = DatasetCache(
       CatsDogsDataSet(TRAIN_SET_FOLDER, max_samples_per_class=None, mode=DataSetMode.Validation),
       transform=validation_transform)
    
    dataloader = DataLoader(dataset, batch_size=200, shuffle=True)
    dataloader_val = DataLoader(dataset_val, batch_size=200, shuffle=True)
    
    net = Network().to(DEVICE)
    loss = torch.nn.CrossEntropyLoss()

    trainer = MonitoredTrainer(net, loss, "model.pt")
    trainer.train(dataloader, dataloader_val)