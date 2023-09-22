import torch
from torch.utils.data import DataLoader
from device import DEVICE
from transform import training_transform, validation_transform
from dataset import CatsDogsDataSet, TRAIN_SET_FOLDER
import json
from network import Network
from trainloop import Trainer
from cache import DatasetCache

class CheckpointTrainer(Trainer):
    def __init__(self, network, loss_function, chkpt_path):
      super().__init__(network, loss_function)

      self.ep = 0
      self.chkpt_path = chkpt_path
      self.best_val_acc = 0
      try:
          chkpt = torch.load(self.chkpt_path)
          self.network.load_state_dict(chkpt["net_state_dict"])
          self.optim.load_state_dict(chkpt["optim_state_dict"])
          self.scheduler.load_state_dict(chkpt["scheduler_state_dict"])
          self.best_val_acc = chkpt["best_val_acc"]
          self.ep = chkpt["epoch"]
      except:
          print("Could not find checkpoint, starting from scratch")

    def train(self, loader_train, loader_val):
      while True:
        train_loss, train_acc = self.epoch(loader_train, True, self.ep)
        val_loss, val_acc = self.epoch(loader_val, False, self.ep)
        self.scheduler.step()
        
        self.ep += 1

        self.logger( {
           "epoch": self.ep,
           "training": { "loss": train_loss, "accuracy": train_acc },
           "validation": { "loss": val_loss, "accuracy": val_acc }
          }
        )

        if val_acc > self.best_val_acc:
          self.best_val_acc = val_acc
          print("Validation accuracy is best, saving checkpoint")
          torch.save({
              "net_state_dict": self.network.state_dict(),
              "optim_state_dict": self.optim.state_dict(),
              "scheduler_state_dict": self.scheduler.state_dict(),
              "best_val_acc": self.best_val_acc,
              "epoch": self.ep
          }, self.chkpt_path)

    def logger(self, statistics):
       print(json.dumps(statistics, indent=3))
    

if __name__ == "__main__":
    dataset = DatasetCache(
      CatsDogsDataSet(TRAIN_SET_FOLDER, max_samples_per_class=600, is_validation=False), 
      transform=training_transform)
    
    dataset_val = DatasetCache(
       CatsDogsDataSet(TRAIN_SET_FOLDER, max_samples_per_class=600, is_validation=True),
       transform=validation_transform)
    
    dataloader = DataLoader(dataset, batch_size=200, shuffle=True)
    dataloader_val = DataLoader(dataset_val, batch_size=200, shuffle=True)
    
    net = Network().to(DEVICE)
    loss = torch.nn.CrossEntropyLoss()

    trainer = CheckpointTrainer(net, loss, "model.pt")
    trainer.train(dataloader, dataloader_val)
        