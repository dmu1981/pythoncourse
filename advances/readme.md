# Python Course - Advances in AI
In diesem Kurs erarbeiten wir gemeinsam ein einfaches PyTorch Skript zum trainieren eines neuronalen Netzwerkes. Dieses soll als Grundlage für weitere Arbeiten und insbesondere ihre eigenen Projekte dienen.

## Daten... wir brauchen Daten!
Registrieren Sie sich bei Kaggle und laden Sie den [Cats vs Dogs](https://www.kaggle.com/competitions/dogs-vs-cats/data) Datensatz herunter. Dieser besteht aus 25.000 Trainingsbilder von Katzen und Hunden. Es geht in diesem Kurs darum mit PyTorch ein Skript zu schreiben welches ein klassisches CNN trainiert um diese beiden Klassen voneinander zu trennen und eine gültige Submission für den Kaggle Contest zu erzeugen die wieder eingereicht werden kann. 

## Das richtige Device finden ([device.py](device.py))
Nicht alle Maschinen sind mit leistungsstarken Grafikkarten ausgestattet. Falls eine GPU vorhanden ist möchten wir diese auch möglichst nutzen, falls jedoch nicht soll die CPU verwendet werden. Wir können das richtige Device einfach über den folgenden Code bestimmen

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print("DEVICE is", DEVICE)




## Die Daten laden ([dataset.py](dataset.py))
PyTorch verwendet [data sets und data loader](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html) um Daten zu organisieren und zu laden. Dabei ist das Data Set dafür zuständig die Daten inklusive Labels (Ground Truth) über eine Listen-API anzubieten. Dazu muß die *\_\_len\_\_* und *\_\_getitem\_\_* Methode implementiert werden. 

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

Das Data Loader ist dafür zuständig aus einem gegebenen Datensatz zufällige Mini-Batches zu laden und diese als Tensor zur Verfügug zu stellen. 

    transform = torch.nn.Sequential(
        transforms.Resize((256, 256)),
    )

    dataset = CatsDogsDataSet(TRAIN_SET_FOLDER, max_samples_per_class=200, transform=transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    batch, label = dataloader.__iter__().__next__()
    batch = torch.Tensor(batch)
    
    grid = make_grid(batch, 8, 2).permute(1,2,0)
    
    plt.imshow(grid)
    plt.show()
 
 ## Transformationen ([transform.py](transform.py))
 Über s.g. [Transforms](https://pytorch.org/vision/stable/transforms.html) können Transformationen mitgegeben werden um die Daten vor bzw. während dem Training zu manipulieren. Damit läßt sich mehr Abwechslung erzeugen um das Training stabiler zu machen und dafür zu Sorgen das das Netzwerk besser generalisiert. In der Regel verwendet man unterschiedliche Transformationen für das Trainings- und das Testset. Daher macht es Sinn die gewünschten Transformationen als Parameter einstellbar zu machen. 

    training_transform = torch.nn.Sequential(
            transforms.Resize((320, 320)),
            transforms.RandomRotation(15),
            transforms.CenterCrop((288, 288)),
            transforms.RandomCrop((256, 256)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ColorJitter(0.3, 0.3, 0.3)
        )

## Das eigentliche Netzwerk ([network.py](network.py))
Wir verwenden eine klassische Netzwerkarchitektur bestehend aus mehreren Faltungslayern gefolgt von Pooling, Batch-Normalisierung und ReLU Aktivierung. Am Ende des Netzes steht ein zweischichtiges Fully-Connected Layer mit 512 Dimensionen in der Mitte sowie DropOut. Der Faltungsteil wird als eigenes PyTorch Module implementiert

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

Das Netzwerk selbst sieht dann so aus

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

Wir können die trainierbaren Parameter des Modells einfach zählen 

    net = Network()
    total_parameters = 0
    model_parameters = filter(lambda p: p.requires_grad, net.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])

    print(f"Network has {params} total parameters")

Einen Mini-Batch durch das Netzwerk vorwärts zu prädizieren ist ebenfalls sehr einfach

    batch, labels = dataloader.__iter__().__next__()
    x = net(batch)
    print(x.shape)

Man beachte die Größe des Ergebnisstensors. In der ersten Dimension gibt es 32 Einträge, einen für jedes Sample im Mini-Batch. In der zweiten Dimension gibt es zwei Werte, dies sind die s.g.  [Logits](https://datascience.stackexchange.com/questions/31041/what-does-logits-in-machine-learning-mean) des Modells. Dabei verwenden wir genau eine Dimension pro zu unterscheidender Klasse, in diesem Fall also genau 2 weil wir ja Katzen von Hunden trennen wollen. Je stärker das erste Neuron feuert, desto wahrscheinlicher handelt es sich bei dem Bild um eine Katze und umgekehrt. Im nächsten Abschnitt werden wir sehen wie wir eine erste Trainingsiteration mit PyTorch durchführen können. 

## Der Trainings-Loop ([trainloop.py](trainloop.py))
Für das Training des Netzwerkes brauchen wir eine Loss-Funktion sowie einen Optimizer um die Parameter anzupassen. Da wir eine binäre Klassifikation durchführen wollen verwenden wir den [Cross Entropy Loss](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html). Als Optimizer wählen wir [Adam](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html) aufgrund seiner guten Konvergenzeigenschaften. 

    net = Network().to(DEVICE)
    optim = torch.optim.Adam(net.parameters(), lr=0.001)
    loss = torch.nn.CrossEntropyLoss()
    
Um eine einzelne Epoche zu trainineren schalten wir das Netzwerk in den Trainingsmodus (z.B. um DropOut zu aktivieren) und iterieren über den ganzen Datensatz

    # We want a dedicated TQDM bar, so we can set the description after each step
    bar = tqdm(dataloader)
    
    # Tell the network whether we are training or evaluating (to disable DropOut)
    if training:
        network.train()
    else:
        network.eval()

    # This epoch starts
    total_loss = 0
    correct = 0
    cnt = 0

    # Iterate over the whole epoch
    for batch, labels in bar:
        # If we are training, zero out the gradients in the network
        if training:
            optim.zero_grad()

Für jeden Mini-Batch führen wir einen Forwardpass durch, berechnen den Loss und aktualisieren unsere Statistiken

    # Do one forward pass
     res = network(batch)

     # Reshape labels for processing
     labels = labels.reshape(-1)

     # Calculcate the (BCE)-Loss
     loss = loss_function(res, labels)

     # Sum the total loss
     total_loss += loss.item()

     # Count how many correct predictions we have (for accuracy)
     correct += torch.sum(torch.argmax(res, dim=1) == labels).item()

     # Count total samples processed
     cnt += batch.shape[0]

     # Update bar description
     bar.set_description(f"loss: {1000.0*total_loss / cnt:.3f}, acc: {100.0*correct/cnt:.2f}%")

Zuletzt propagieren wir die Gradienten rückwärts durchs Netz und dann aktualisieren die Parameter mit dem Optimizer
     
     # If we are training, do backward pass 
     if training:
        # Calculate backward gradients
        loss.backward()

        # Step the optimizer
        optim.step()

## Checkpoints ([checkpoints.py](checkpoints.ps))
Um das Training nicht immer wieder von vorne beginnen zu müssen macht es Sinn den aktuellen Zustand des Netzwerkes zu speichern. In PyTorch kann man dafür s.g. Checkpoints anlegen. Um den aktuellen Stand vollständig zu speichern muß neben dem Netzwerk selbst auch noch das Zustand des Optimizers gespeichert werden. 

    for i in range(20):
        epoch(dataloader, net, optim, loss, True, ep)
        ep += 1

        torch.save({
            "net_state_dict": net.state_dict(),
            "optim_state_dict": optim.state_dict(),
            "epoch": ep
        }, "model.pt")

Das Laden eines Checkpoints von der Festplatte geschieht über die *torch.load* Funktion. Diese kann natürlich scheitern, z.B. falls gar kein Checkpoint existiert. Über einen entsprechenden try/except Block kann in diesem Fall einfach elegant ein neues Training gestartet werden.

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

## Das Training überwachen ([monitor.py](monitor.py))
Installieren und starten Sie tensorpoard

    pip install tensorboard
    tensorboard --logdir=runs

Öffnen Sie http://localhost:6006/ in ihrem Browser um auf das TensorBoard zuzugreifen. 

