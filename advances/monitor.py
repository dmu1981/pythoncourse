import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from device import DEVICE
from transform import training_transform, validation_transform
from dataset import CatsDogsDataSet, TRAIN_SET_FOLDER, DataSetMode
from cache import DatasetCache
import numpy as np
from network import Network, Down
from checkpoints import CheckpointTrainer

def weight_histograms_conv2d(writer, step, weights, prefix):
  #weights_shape = weights.shape
  #num_kernels = weights_shape[0]

  writer.add_histogram(f"{prefix}/conv", weights.flatten(), global_step=step, bins='tensorflow')

  #for k in range(num_kernels):
   # flattened_weights = weights[k].flatten()
    #tag = f"{prefix}/kernel_{k}"
    #writer.add_histogram(tag, flattened_weights, global_step=step, bins='tensorflow')


def weight_histograms_linear(writer, step, weights, prefix):
  flattened_weights = weights.flatten()
  tag = f"{prefix}"
  writer.add_histogram(tag, flattened_weights, global_step=step, bins='tensorflow')


def weight_histograms(writer, step, model, prefix):
  #print("Visualizing model weights...")
  # Iterate over all model layers
  for layer_number, layer in enumerate(model.children()):
    # Compute weight histograms for appropriate layer
    #print(layer)
    if isinstance(layer, nn.Sequential):
       weight_histograms(writer, step, layer, prefix=prefix)

    if isinstance(layer, Down):
       weight_histograms(writer, step, layer.children().__iter__().__next__(), prefix=f"down_{layer_number}")

    if isinstance(layer, nn.Conv2d):
      weights = layer.weight
      weight_histograms_conv2d(writer, step, weights, prefix=f"{prefix}/layer_{layer_number}")
    elif isinstance(layer, nn.Linear):
      weights = layer.weight
      weight_histograms_linear(writer, step, weights, prefix=f"{prefix}/layer_{layer_number}")

class MonitoredTrainer(CheckpointTrainer):
    def __init__(self, network, loss_function, chkpt_path):
        super().__init__(network, loss_function, chkpt_path)

        self.writer = SummaryWriter()

    def logger(self, statistics):
        self.writer.add_scalar("train/loss", statistics["training"]["loss"], statistics["epoch"])
        self.writer.add_scalar("train/accuracy", statistics["training"]["accuracy"], statistics["epoch"])
        self.writer.add_scalar("validation/loss", statistics["validation"]["loss"], statistics["epoch"])
        self.writer.add_scalar("validation/accuracy", statistics["validation"]["accuracy"], statistics["epoch"])

        weight_histograms(self.writer, statistics["epoch"], self.network, prefix="")
    
    def report_top_loss(self, top_loss_samples, epoch):
        top_loss_samples = top_loss_samples[:64,:,:]
        img = make_grid(top_loss_samples, nrow=8)#.permute(2,1,0)
        self.writer.add_image("top_loss_samples", img, global_step=epoch)
        return None
    
if __name__ == "__main__":
    dataset = DatasetCache(
      CatsDogsDataSet(TRAIN_SET_FOLDER, max_samples_per_class=2000, mode=DataSetMode.Final), 
      transform=training_transform)
    
    dataset_val = DatasetCache(
       CatsDogsDataSet(TRAIN_SET_FOLDER, max_samples_per_class=2000, mode=DataSetMode.Validation),
       transform=validation_transform)
    
    dataloader = DataLoader(dataset, batch_size=400, shuffle=True)
    dataloader_val = DataLoader(dataset_val, batch_size=400, shuffle=True)
    
    net = Network().to(DEVICE)
    loss = torch.nn.CrossEntropyLoss(reduction="none")

    trainer = MonitoredTrainer(net, loss, "model.pt")
    trainer.train(dataloader, dataloader_val)