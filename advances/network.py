import torch
from torch.utils.data import DataLoader
from device import DEVICE
from transform import training_transform
from dataset import CatsDogsDataSet, TRAIN_SET_FOLDER
import numpy as np

class Down(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super(Down, self).__init__()
        
        self.seq  = torch.nn.Sequential(
            torch.nn.Conv2d(in_features, out_features, kernel_size=(3,3), padding="same"),
            torch.nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
            torch.nn.BatchNorm2d(num_features = out_features),
            torch.nn.ReLU()
            )

    def forward(self, x):
        return self.seq(x)

class Network(torch.nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        self.seq = torch.nn.Sequential(
            Down(in_features =   3, out_features =   8), #   8 x 128 x 128
            Down(in_features =   8, out_features =  16), #  16 x  64 x  64
            Down(in_features =  16, out_features =  32), #  32 x  32 x  32 
            Down(in_features =  32, out_features =  64), #  64 x  16 x  16 
            Down(in_features =  64, out_features = 128), # 128 x   8 x   8
            Down(in_features = 128, out_features = 256), # 256 x   4 x   4
            torch.nn.Flatten(), # 4096 dimensional
            torch.nn.Linear(4096, 512), # 512 dimensional
            torch.nn.ReLU(), # Another ReLU
            torch.nn.Dropout(0.5),
            torch.nn.Linear(512, 2) # Go down to two neurons, one for cats, one for dogs
        )

    def forward(self, x):
        return self.seq(x)

if __name__ == "__main__":
    dataset = CatsDogsDataSet(TRAIN_SET_FOLDER, max_samples_per_class=2000, transform=training_transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    net = Network().to(DEVICE)
    total_parameters = 0
    model_parameters = filter(lambda p: p.requires_grad, net.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])

    print(f"Network has {params} total parameters")

    batch, labels = dataloader.__iter__().__next__()
    x = net(batch)
    print(x.shape)

