import torch
from torch import nn
import torchvision
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from tqdm import tqdm

dataset = torchvision.datasets.FashionMNIST("fashionMNIST", 
                                            download=True,
                                            train=True,
                                            transform=torchvision.transforms.ToTensor())

loader = DataLoader(dataset, batch_size=16, shuffle=True)

dataset_test = torchvision.datasets.FashionMNIST("fashionMNIST", 
                                            download=True,
                                            train=False,
                                            transform=torchvision.transforms.ToTensor())

loader_test = DataLoader(dataset_test, batch_size=16, shuffle=True)


batch, labels = loader.__iter__().__next__()
grid = torchvision.utils.make_grid(batch, 4).permute(1,2,0)
plt.imshow(grid)
plt.show()

# Aufgabe 1
# In dieser Übung wollen wir versuchen die Bilder aus dem CIFAR10 Datensatz
# mit Hilfe der tradionellen voll-vernetzten Netzwerkarchitektur aus der Vorlesung
# zu klassifizieren. CIFAR 10 besteht aus 50.000 Bildern aus 10 Klassen.
class FullNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.flatten = nn.Flatten()
        self.sigmoid = nn.Sigmoid()

        # TODO:
        # Erzeugen Sie hier mit Hilfe des nn.Linear Moduls
        # drei Schichten mit sinnvoller Eingabe- und Ausgabegröße
        # Hinweis: Die Bild-Daten haben eine Auflösung von 28x28 Pixeln mit 1 Kanalen
        # Nach dem flatten entspricht dies 28*28 = 784 Eingabedimensionen
        # Verwenden Sie noch 392 Neuronen in der ersten und 128 Neuronen in der zweiten
        # Schicht. Auf der letzten Schicht brauchen Sie exakt 10 Neuronen (für jede Klasse 1)
        

    def forward(self, x):
        # TODO: 
        # Implementieren Sie den Forward-Pass ihres Netzwerkes
        # indem Sie die Daten zunächst flatten und dann suksezive 
        # durch die linearen Schichten und den Sigmoid geben. 
        # ACHTUNG: Auf der letzten Schicht brauchen Sie keine Nicht-Linearität

        return x

# Aufgabe 2
# Wir wollen nun eine klassische Faltungsarchitektur ausprobieren.
# HINWEIS:
# Denken Sie daran das sie weiter unten im Code ihr ConvNet instantieren
# müssen anstatt dem FullNet von oben
class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        
        # TODO: 
        # Erzeugen Sie mittels nn.MaxPool2D und nn.Conv2d 
        # geeigente Layer. Sie benötigen 2 Faltungen mit je einer kernel_size von (5,5)
        # Verwenden Sie padding="same". Da sie nach jeder Faltung eine MaxPooling anwenden
        # werden vergrößern Sie gleichzeitig die Anzahl Kanäle (also z.B. 8, dann 16)
        

        # TODO:
        # Überlegen Sie wieviele "Neuronen" ihr Netzwerk nach den drei Faltungslayern
        # noch hat (bei 16 Kanälen) und erzeugen Sie ein entsprechend großes Fully-Connected
        # Layer mit nn.Linear (wie oben))
        
        
    def forward(self, x):
        # TODO:
        # Implementieren Sie den Forward-Pass ihres Faltungsnetzwerkes
        # indem Sie immer abwechseln zwischen Faltung und Pooling + ReLU. 
        

        # Flatten Sie nun das Ergebniss und wenden Sie den letzten Fully-Connected Teil
        # an um ihr Ergebniss zu erzeugen
        
        return x

net = FullNet()
optim = torch.optim.Adam(net.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()



### AB HIER BRAUCHEN SIE NICHTS ZU TUN AUSSER ZU VERSTEHEN WAS PASSIERT
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






