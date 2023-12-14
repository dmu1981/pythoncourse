import torch
from torch import nn
import torchvision
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

dataset = torchvision.datasets.CIFAR10("cifar10", 
                                       download=True,
                                       train=True,
                                       transform=torchvision.transforms.ToTensor())

loader = DataLoader(dataset, batch_size=64, shuffle=True)

dataset_test = torchvision.datasets.CIFAR10("cifar10", 
                                       download=True,
                                       train=False,
                                       transform=torchvision.transforms.ToTensor())

loader_test = DataLoader(dataset_test, batch_size=64, shuffle=True)

batch, labels = loader.__iter__().__next__()
grid = torchvision.utils.make_grid(batch, 8).permute(1,2,0)
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
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)) 
        self.c1 = nn.Conv2d( 3,  8, kernel_size=(5,5), padding="same")    
        self.c2 = nn.Conv2d( 8, 16, kernel_size=(5,5), padding="same")
        self.c3 = nn.Conv2d(16, 32, kernel_size=(5,5), padding="same")
        self.linear = nn.Linear(512, 10)

    def forward(self, x):
        # TODO:
        # Implementieren Sie den Forward-Pass ihres Faltungsnetzwerkes
        # ähnlich zu Aufgabe 1
        x1 = self.relu(self.pool(self.c1(x)))
        x2 = self.relu(self.pool(self.c2(x1)))
        x3 = self.relu(self.pool(self.c3(x2)))
        x = self.linear(self.flatten(x3))
        return x, x1, x2, x3



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

        out, c1, c2, c3 = net(batch)
        loss = criterion(out, labels)
        loss.backward()

        

        total_correct += torch.sum(torch.argmax(out, dim=1) == labels)
        total_loss += loss.item()
        total_cnt += batch.shape[0]

        bar.set_description(f"train: epoch={epoch}, loss={1000.0*total_loss / total_cnt:.3f}, acc={total_correct / total_cnt * 100:.2f}%")

        optim.step()

    batch, _ = loader.__iter__().__next__()
    out, c1, c2, c3 = net(batch)
    fix, axs = plt.subplots(1,4)
    c1 = c1[0].view( 8,1,16,16).repeat([1,3,1,1])
    c2 = c2[0].view(16,1, 8, 8).repeat([1,3,1,1])
    c3 = c3[0].view(32,1, 4, 4).repeat([1,3,1,1])
    grid1 = torchvision.utils.make_grid(c1.detach().cpu(), 1).permute(1,2,0)
    grid2 = torchvision.utils.make_grid(c2.detach().cpu(), 2).permute(1,2,0)
    grid3 = torchvision.utils.make_grid(c3.detach().cpu(), 2).permute(1,2,0)
    axs[0].imshow(batch[0].permute(1,2,0))
    axs[1].imshow(grid1[:,:,0], cmap="jet")
    axs[2].imshow(grid2[:,:,0], cmap="jet")
    axs[3].imshow(grid3[:,:,0], cmap="jet")
    plt.show()
    
    total_loss = 0
    total_cnt = 0
    total_correct = 0

    bar = tqdm(loader_test)
    for batch, labels in bar:
        with torch.no_grad():
            out, c1, c2, c3 = net(batch)
            loss = criterion(out, labels)

        total_correct += torch.sum(torch.argmax(out, dim=1) == labels)
        total_loss += loss.item()
        total_cnt += batch.shape[0]

        bar.set_description(f"test: epoch={epoch}, loss={1000.0*total_loss / total_cnt:.3f}, acc={total_correct / total_cnt * 100:.2f}%")