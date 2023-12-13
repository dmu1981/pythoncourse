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

loader = DataLoader(dataset, batch_size=256, shuffle=True)

dataset_test = torchvision.datasets.CIFAR10("cifar10", 
                                       download=True,
                                       train=False,
                                       transform=torchvision.transforms.ToTensor())

loader_test = DataLoader(dataset_test, batch_size=256, shuffle=True)

batch, labels = loader.__iter__().__next__()
grid = torchvision.utils.make_grid(batch, 4).permute(1,2,0)
plt.imshow(grid)
plt.show()

# Aufgabe 1
# Wir wollen nun unser neuronales Netzwerk modularisieren.
# Dazu implementieren wir zunächst ein "DownConv"-Modul.
# Hier möchten wir das Eingabebild zweimal falten bevor wir poolen.
# Zwischen beiden Faltungen wenden wir auch wieder eine ReLU als nicht-linearität an.
#
# Dieses Modul soll also die folgende Operation abbilden
#
#      x -> Faltung1 -> ReLU -> Faltung2 -> MaxPool -> ReLU
#
class DownConv(nn.Module):
    def __init__(self, input_channels, intermediate_channels, output_channels, kernel_size = (5,5)):
        super().__init__()

        # TODO:
        # Erzeugen Sie geeignete Layer (ReLU, Pooling, 2x Faltung)
        # um das Eingabebild (mit C=input_channels) auf (C=intermediate_channels) zu falten,
        # und das Zwischenbild auf (C=output_channels) zu falten.
        #
        # TODO:
        # Als Bonus können Sie vor jeder ReLU ein BatchNorm2d Layer einfügen
        #   https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html
        
    def forward(self, x):
        # TODO:
        # Führen Sie die Mehrfach-Faltung wie oben beschrieben aus und geben Sie das Ergebniss zurück
        return x

# Aufgabe 2
# Verwenden Sie nun drei von Ihnen implementierte DownConv Module um das Eingangsbild (3x32x32)
# auf (32x4x4) zu falten. Wählen Sie geeignete intermediate_channels
class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
                
        # TODO: 
        # Erzeugen Sie geeignete Layer 
        
    def forward(self, x):
        # TODO:
        # Implementieren Sie den Forward-Pass ihres Faltungsnetzwerkes
        # indem Sie ihre oben erzeugen DownConv Module aufrufen
        return x

# Aufgabe 3:    
# Implementieren Sie die load_checkpoint and save_checkpoint Methoden.
#
#   https://pytorch.org/tutorials/beginner/saving_loading_models.html
#   https://pytorch.org/tutorials/recipes/recipes/what_is_state_dict.html
def load_checkpoint(net, optim):
    # Hinweis: Da das laden eines Checkpoints aus verschiedenen Gründen fehlschlagen kann
    # macht es Sinn dies in einem Try-Except Block zu tun
    try:
        # TODO:
        # Laden sie die "model.pt" Datei und daraus die Modell- und Optimizerparameter
        #
        #   https://pytorch.org/docs/stable/generated/torch.load.html
        #
        pass
    except:
        print("Could not load checkpoint, starting from scratch")

def save_checkpoint(net, optim):
    # TODO:
    # Speichern die Modell- und Optimizerparameter in der "model.pt"-Datei
    #
    #   https://pytorch.org/docs/stable/generated/torch.save.html#torch.save
    #
    pass

### AB HIER BRAUCHEN SIE NICHTS ZU TUN AUSSER ZU VERSTEHEN WAS PASSIERT
net = ConvNet()
optim = torch.optim.Adam(net.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

load_checkpoint(net, optim)

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

    save_checkpoint(net, optim)



