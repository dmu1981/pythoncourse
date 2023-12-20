import torch
from torch import nn
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Aufgabe 1:
# Erzeugen Sie wie in der Vorlesung gezeigt einen (globalen!) SummaryWriter 
# und implementieren Sie die "writeBatchSummary"-Methode um zu jedem Batch 
# den erreichten Loss und die Erreichte Genauigkeit ins TensorBoard zu schreiben
def writeBatchSummary(predictions, labels, criterion, step):
    pass



# Aufgabe 2:
# Erweitern Sie die "writeBatchSummary"-Methode derart, dass Sie über 25000 Samples eine Statistik sammeln
# bevor Sie Daten ans TensorBoard schicken. 



# Aufgabe 3:
# Berechnen Sie zusätzlich zur allgemeinen Accuracy auch die 
# "Confusion-Matrix", also die Verwechslungen zwischen allen Klassenpaaren. Verwenden Sie die SeaBorn-Library
# und erzeugen Sie eine s.g. "HeatMap" wo sie die Verwechslungen darstellen
#
#   https://seaborn.pydata.org/examples/spreadsheet_heatmap.html
#
# Senden Sie die entstehende Figure dann als TensorBoard
#
#   https://pytorch.org/docs/stable/tensorboard.html
#        









dataset = torchvision.datasets.CIFAR10("cifar10", 
                                       download=True,
                                       train=True,
                                       transform=torchvision.transforms.ToTensor())

loader = DataLoader(dataset, batch_size=256, shuffle=True)

dataset_test = torchvision.datasets.CIFAR10("cifar10", 
                                       download=True,
                                       train=False,
                                       transform=torchvision.transforms.ToTensor())

loader_test = DataLoader(dataset_test, batch_size=256, shuffle=True)

batch, labels = loader.__iter__().__next__()
grid = torchvision.utils.make_grid(batch, 16).permute(1,2,0)

class DownConv(nn.Module):
    def __init__(self, input_channels, intermediate_channels, output_channels, kernel_size = (5,5)):
        super().__init__()

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.conv1 = nn.Conv2d(input_channels, intermediate_channels, kernel_size=kernel_size, padding="same")
        self.conv2 = nn.Conv2d(intermediate_channels, output_channels, kernel_size=kernel_size, padding="same")
        self.bn1 = nn.BatchNorm2d(intermediate_channels)
        self.bn2 = nn.BatchNorm2d(output_channels)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.pool(self.conv2(x))))
        return x

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
                
        self.down1 = DownConv( 3, 16, 16)
        self.down2 = DownConv(16, 32, 32)
        self.down3 = DownConv(32, 64, 64)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(1024, 10)
        
    def forward(self, x):
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.linear(self.flatten(x))
        return x

def load_checkpoint(net, optim):
    try:
        chkpt = torch.load("model.pt")
        net.load_state_dict(chkpt["model"])
        optim.load_state_dict(chkpt["optim"])
        return chkpt["step"]
    except:
        print("Could not load checkpoint, starting from scratch")
        return 0

def save_checkpoint(net, optim, global_step):
    torch.save({
        "model": net.state_dict(),
        "optim": optim.state_dict(),
        "step": global_step
    }, "model.pt")
    pass

net = ConvNet()
optim = torch.optim.Adam(net.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

global_step = load_checkpoint(net, optim)

for epoch in range(10):
    bar = tqdm(loader)
    
    for batch, labels in bar:
        optim.zero_grad()

        out = net(batch)
        loss = criterion(out, labels)
        loss.backward()

        writeBatchSummary(out, labels, criterion, global_step)
    
        optim.step()

        global_step += 1
    
    save_checkpoint(net, optim, global_step)
    



