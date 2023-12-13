import torch
from torch import nn
import torchvision
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from tqdm import tqdm

dataset = torchvision.datasets.CIFAR10("cifar10", 
                                       download=True,
                                       train=True,
                                       transform=torchvision.transforms.ToTensor())

loader = DataLoader(dataset, batch_size=16, shuffle=True)

dataset_test = torchvision.datasets.CIFAR10("cifar10", 
                                       download=True,
                                       train=False,
                                       transform=torchvision.transforms.ToTensor())

loader_test = DataLoader(dataset_test, batch_size=16, shuffle=True)

batch, labels = loader.__iter__().__next__()
grid = torchvision.utils.make_grid(batch, 4).permute(1,2,0)
plt.imshow(grid)
plt.show()

# Aufgabe 1
# Passen Sie nun den Code aus Aufgabe 1 an das CIFAR10 Datenset an.
# CIFAR 10 besteht aus 50.000 Farbbildern der Größe 32x32 
class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()

        # TODO: 
        # Erzeugen Sie geeignete Layer um das Eingangsbild (3x32x32) auf
        # die geeignete Tensorgröße herunterzubrechen.
        # HINWEIS: CIFAR10 hat 10 Klassen, sie brauchen demnach 10 Neuronen am Ende.
        
    def forward(self, x):
        # TODO:
        # Implementieren Sie den Forward-Pass ihres Faltungsnetzwerkes
        # ähnlich zu Aufgabe 1
        return x
    


### AB HIER BRAUCHEN SIE NICHTS ZU TUN AUSSER ZU VERSTEHEN WAS PASSIERT
net = ConvNet()
optim = torch.optim.Adam(net.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(10):
    total_loss = 0
    total_cnt = 0
    total_correct = 0
    bar = tqdm(loader)
    for batch, labels in bar:
        optim.zero_grad()

        out = net(batch)
        loss = criterion(out, labels)
        loss.backward()
        
        total_correct += torch.sum(torch.argmax(out, dim=1) == labels)
        total_loss += loss.item()
        total_cnt += batch.shape[0]

        bar.set_description(f"train: epoch={epoch}, loss={1000.0*total_loss / total_cnt:.3f}, acc={total_correct / total_cnt * 100:.2f}%")

        optim.step()

    total_loss = 0
    total_cnt = 0
    total_correct = 0

    bar = tqdm(loader_test)
    for batch, labels in bar:
        with torch.no_grad():
            out = net(batch)
            loss = criterion(out, labels)
        
        total_correct += torch.sum(torch.argmax(out, dim=1) == labels)
        total_loss += loss.item()
        total_cnt += batch.shape[0]

        bar.set_description(f"test: epoch={epoch}, loss={1000.0*total_loss / total_cnt:.3f}, acc={total_correct / total_cnt * 100:.2f}%")




