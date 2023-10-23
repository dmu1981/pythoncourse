# Python Course - Advances in AI
In diesem Kurs erarbeiten wir gemeinsam ein einfaches PyTorch Skript zum trainieren eines neuronalen Netzwerkes. Dieses soll als Grundlage für weitere Arbeiten und insbesondere ihre eigenen Projekte dienen.

## Daten... wir brauchen Daten!
Registrieren Sie sich bei Kaggle und laden Sie den [Cats vs Dogs Redux: Kernels Edition](https://www.kaggle.com/competitions/dogs-vs-cats-redux-kernels-edition/overview/evaluation) Datensatz herunter. Dieser besteht aus 25.000 Trainingsbilder von Katzen und Hunden. Es geht in diesem Kurs darum mit PyTorch ein Skript zu schreiben welches ein klassisches CNN trainiert um diese beiden Klassen voneinander zu trennen und eine gültige Submission für den Kaggle Contest zu erzeugen die wieder eingereicht werden kann. 

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
        transforms.Resize((128, 128), antialias=True),
        transforms.ConvertImageDtype(torch.float32)
    )

    dataset = CatsDogsDataSet(TRAIN_SET_FOLDER, max_samples_per_class=200, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    batch, label = dataloader.__iter__().__next__()
    batch = torch.Tensor(batch)
    
    grid = make_grid(batch, 8, 2).permute(1,2,0)
    
    plt.imshow(grid.cpu())
    plt.show()
 
 ## Transformationen ([transform.py](transform.py))
 Über s.g. [Transforms](https://pytorch.org/vision/stable/transforms.html) können Transformationen mitgegeben werden um die Daten vor bzw. während dem Training zu manipulieren. Damit läßt sich mehr Abwechslung erzeugen um das Training stabiler zu machen und dafür zu Sorgen das das Netzwerk besser generalisiert. In der Regel verwendet man unterschiedliche Transformationen für das Trainings- und das Testset. Daher macht es Sinn die gewünschten Transformationen als Parameter einstellbar zu machen. 

    training_transform = transforms.Compose([
        transforms.RandomRotation(15),
        transforms.CenterCrop((144, 144)),
        transforms.RandomCrop((128, 128)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ColorJitter(0.3, 0.3, 0.3),
        transforms.ConvertImageDtype(torch.float32)
        ]
    )t

## Das eigentliche Netzwerk ([network.py](network.py))
Wir verwenden eine klassische Netzwerkarchitektur bestehend aus mehreren Faltungslayern gefolgt von Pooling, Batch-Normalisierung und ReLU Aktivierung. Am Ende des Netzes steht ein zweischichtiges Fully-Connected Layer mit 512 Dimensionen in der Mitte sowie DropOut. Der Faltungsteil wird als eigenes PyTorch Module implementiert

    class Down(torch.nn.Module):
        def __init__(self, in_features, out_features):
            super(Down, self).__init__()
            
            self.seq  = torch.nn.Sequential(
                torch.nn.Conv2d(in_features, out_features, kernel_size=(5,5), padding="same"),
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
            Down(in_features =   3, out_features =  32), #  32x64x64
            torch.nn.Dropout2d(0.2), # Compare https://arxiv.org/abs/1411.4280
            Down(in_features =  32, out_features =  64), #  64x32x32
            torch.nn.Dropout2d(0.2),
            Down(in_features =  64, out_features =  128), #  128x16x16
            torch.nn.Dropout2d(0.2),
            Down(in_features =  128, out_features =  256), #  256x8x8
            torch.nn.Dropout2d(0.2),
            torch.nn.Flatten(), # 4096 dimensional
            torch.nn.Linear(16384, 512), # 512 dimensional
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
    optim = torch.optim.Adam(net.parameters(), lr=0.0001)
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
Installieren und starten Sie tensorboard

    pip install tensorboard
    tensorboard --logdir=runs

Öffnen Sie http://localhost:6006/ in ihrem Browser um auf das TensorBoard zuzugreifen. 

Hinweis:
  Es kann sein das Sie die SETUPTOOLS von Python downgraden müssen (vgl. https://github.com/pytorch/pytorch/pull/69904)

    pip install setuptools==59.5.0



Um das interkative Tensorboard mit Daten zu befüllen erzeugen wir einen s.g. SummaryWriter

    self.writer = SummaryWriter()

und schreiben dann unsere Skalaren Meßgrößen über diesen Writer auf die Festplatte

    def logger(self, statistics):
        self.writer.add_scalar("train/loss", statistics["training"]["loss"], statistics["epoch"])
        self.writer.add_scalar("train/accuracy", statistics["training"]["accuracy"], statistics["epoch"])
        self.writer.add_scalar("validation/loss", statistics["validation"]["loss"], statistics["epoch"])
        self.writer.add_scalar("validation/accuracy", statistics["validation"]["accuracy"], statistics["epoch"])

Der TensorBoard Service überwacht den Ordner und lädt Änderungen on-the-fly nach. Die Werte werden dann in interaktive Graphen im Browser dargestellt. 

## Profiling ([profiling.py](profiling.py))
PyTorch kommt mit einem eigenen Profiler. Damit läßt sich überprüfen wo die Rechenzeit (sowohl auf CPU als auch auf der GPU) investiert wird.

    start = time.time()
    batch, labels = dataloader.__iter__().__next__()
    with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU], record_shapes=True) as prof:
      with record_function("model_inference"):
        res = net(batch)
        
    end = time.time()
    print(end - start)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

Das Ergebniss kann dann z.B. so aussehen:

---------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------   
                             Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls  
---------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------   
                  model_inference         0.56%       5.683ms        98.30%     997.079ms     997.079ms      51.000us         0.00%        1.021s        1.021s             1   
                     aten::conv2d         0.01%      58.000us        80.14%     812.916ms     203.229ms      11.000us         0.00%     843.412ms     210.853ms             4
          aten::_convolution_mode         0.34%       3.458ms        80.13%     812.858ms     203.214ms       7.000us         0.00%     843.401ms     210.850ms             4
               aten::_convolution         0.31%       3.187ms        79.79%     809.400ms     202.350ms      28.000us         0.00%     843.394ms     210.849ms             4
          aten::cudnn_convolution        79.30%     804.431ms        79.46%     806.021ms     201.505ms     839.392ms        82.18%     839.401ms     209.850ms             4
            aten::feature_dropout         0.05%     540.000us         9.46%      95.921ms      23.980ms      41.000us         0.00%     103.018ms      25.755ms             4
                 aten::bernoulli_         9.38%      95.147ms         9.38%      95.147ms      23.787ms     101.942ms         9.98%     101.942ms      25.485ms             4
                 aten::batch_norm         0.00%      41.000us         7.51%      76.195ms      19.049ms      12.000us         0.00%      69.808ms      17.452ms             4
     aten::_batch_norm_impl_index         1.23%      12.456ms         7.51%      76.154ms      19.038ms      10.000us         0.00%      69.796ms      17.449ms             4
           aten::cudnn_batch_norm         6.24%      63.294ms         6.28%      63.698ms      15.925ms      69.762ms         6.83%      69.786ms      17.447ms             4
---------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
Self CPU time total: 1.014s
Self CUDA time total: 1.021s

## Auf Kaggle die Ergebnisse einreichnen ([submission.py](submission.py))

Um die Ergebnisse auf Kaggle einreichen zu können und einen offiziellen Score zu bekommen müssen wir für jedes Bild aus dem Testset eine Prädiktion berechnen und die geschätzte Wahrscheinlichkeit für die Zugehörigkeit zur Hundeklasse berechnen. Wir erzeugen also zunächst das Dataset und den dazugehörigen DataLoader. Wir verwenden natürlich die Validation Transform weil wir möglichst wenig Veränderung an den Bildern zulassen wollen. 

    dataset = CatsDogsDataSet(
        TEST_SET_FOLDER, max_samples_per_class=None, mode=DataSetMode.Test, transform=validation_transform) 
    
    dataloader = DataLoader(dataset, batch_size=200, shuffle=False)
    
Nun laden wir das Netzwerk aus dem CheckPoint und schalten in den evaluations modus um DropOut zu deaktivieren.     
    net = Network().to(DEVICE)
    net.eval()
    chkpt = torch.load("model.pt")
    net.load_state_dict(chkpt["net_state_dict"])

Wir beginnen mit einem vorausgefüllten Tensor für die Prädiktionen

    predictions = torch.zeros(len(dataset)).to(DEVICE)

Wir bauchen in diesem Schritt keine Gradienten, können die Berechnung selbiger also deaktivieren.
    
    with torch.no_grad():
      for batch, labels in tqdm(dataloader):
          labels = labels.reshape(-1)
          prediction = torch.softmax(net(batch), dim=1)[:,1]
          predictions[labels] = prediction

Wir wenden den SoftMax auf die Logits an und verwenden den Eintrag in der Hundeklasse als geschätzte Wahrscheinlichkeit für das Modell

Python hat eine eigene Library zum schreiben von CSV Dateien. Damit schreiben wir unsere Prädiktionen in eine Datei

    with open("submission.csv", "w", newline='') as f:
      writer = csv.writer(f)
      writer.writerow(["id", "label"])
      for index, prediction in enumerate(predictions):
         writer.writerow([index+1, prediction.item()])

Diese Datei können wir nun auf Kaggle hochladen und erhalten einen offiziellen Score für unsere Ergebnisse.

## Challenge
Wer erreicht den niedrigsten offiziellen Kaggle-Score?