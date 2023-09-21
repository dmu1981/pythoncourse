import torch
from torch.utils.data import DataLoader
from device import DEVICE
from transform import training_transform_64
from dataset import CatsDogsDataSet, TRAIN_SET_FOLDER
import numpy as np
from network import Network
from trainloop import epoch

if __name__ == "__main__":
    dataset = CatsDogsDataSet(TRAIN_SET_FOLDER, max_samples_per_class=1280, transform=training_transform_64)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
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
        epoch(dataloader, net, optim, loss, True, ep)
        ep += 1

        torch.save({
            "net_state_dict": net.state_dict(),
            "optim_state_dict": optim.state_dict(),
            "epoch": ep
        }, "model.pt")