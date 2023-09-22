import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from device import DEVICE
from transform import training_transform, validation_transform
from dataset import CatsDogsDataSet, TRAIN_SET_FOLDER
import numpy as np
from network import Network
from trainloop import epoch



if __name__ == "__main__":
    writer = SummaryWriter()

    dataset = CatsDogsDataSet(TRAIN_SET_FOLDER, max_samples_per_class=None, transform=training_transform, is_validation=False)
    dataset_val = CatsDogsDataSet(TRAIN_SET_FOLDER, max_samples_per_class=None, transform=validation_transform, is_validation=True)
    
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
    dataloader_val = DataLoader(dataset_val, batch_size=256, shuffle=True)
    
    net = Network().to(DEVICE)
    optim = torch.optim.Adam(net.parameters(), lr=0.0001)
    loss = torch.nn.CrossEntropyLoss()
    ep = 0
    try:
        chkpt = torch.load("model.pt")
        net.load_state_dict(chkpt["net_state_dict"])
        optim.load_state_dict(chkpt["optim_state_dict"])        
        ep = chkpt["epoch"]
    except:
        print("Could not find checkpoint, starting from scratch")

    for i in range(2000):
        ls, acc = epoch(dataloader, net, optim, loss, True, ep)
        writer.add_scalar("train/loss", ls, ep)
        writer.add_scalar("train/acc", acc, ep)
        
        ls, acc = epoch(dataloader_val, net, optim, loss, False, ep)
        writer.add_scalar("val/loss", ls, ep)
        writer.add_scalar("val/acc", acc, ep)

        ep += 1

        torch.save({
            "net_state_dict": net.state_dict(),
            "optim_state_dict": optim.state_dict(),
            "epoch": ep
        }, "model.pt")