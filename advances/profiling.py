import torch
from torch.utils.data import DataLoader
from device import DEVICE
from transform import training_transform
from dataset import CatsDogsDataSet, TRAIN_SET_FOLDER
import numpy as np
from network import Network
from tqdm import tqdm
from torch.profiler import profile, record_function, ProfilerActivity
import time

if __name__ == "__main__":
    dataset = CatsDogsDataSet(TRAIN_SET_FOLDER, max_samples_per_class=200, transform=training_transform)
    dataloader = DataLoader(dataset, batch_size=200, shuffle=True)
    
    net = Network().to(DEVICE)
    iterator = dataloader.__iter__()
  
    start = time.time()
    batch, labels = dataloader.__iter__().__next__()
    with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU], record_shapes=True) as prof:
      with record_function("model_inference"):
        res = net(batch)
        
    end = time.time()
    print(end - start)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))