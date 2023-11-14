import torch
from torch import nn
import data
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device is {DEVICE}")

BATCH_SIZE = 128

class Net(nn.Module):
  def __init__(self):
    super().__init__()
    
    self.down1 = nn.Sequential(
      nn.Conv2d(14,  16, kernel_size=(9, 1), padding=(4,0)),
      nn.MaxPool2d(kernel_size=(2,1), stride=(2,1)),
      nn.GELU(),
    )

    self.down2 = nn.Sequential(
      nn.Conv2d(16,  32, kernel_size=(9, 1), padding=(4,0)),
      nn.MaxPool2d(kernel_size=(2,1), stride=(2,1)),
      nn.GELU(),
    )

    self.down3 = nn.Sequential(
      nn.Conv2d(32,  64, kernel_size=(9, 1), padding=(4,0)),
      nn.MaxPool2d(kernel_size=(2,1), stride=(2,1)),
      nn.GELU(),
    )

    self.transformer = nn.Sequential(
      nn.TransformerEncoderLayer(d_model=64, nhead=8, batch_first=True),
      nn.TransformerEncoderLayer(d_model=64, nhead=8, batch_first=True),
    )

    self.mlp = nn.Sequential(
      nn.Linear(64, 256),
      nn.GELU(),
      nn.Linear(256, 2)
    )

  def forward(self, x):
    x = torch.unsqueeze(torch.transpose(x, dim0=2, dim1=1), 3)
    
    x = self.down1(x)
    x = self.down2(x)
    x = self.down3(x)

    x = x.squeeze(3).transpose(dim0=1,dim1=2)

    x = self.transformer(x)

    return self.mlp(x[:,0,:])
    #return self.mlp(x)
  
eegdataset = data.EEGDataSet(seqlen=80)
dataloader = torch.utils.data.DataLoader(eegdataset, batch_size=BATCH_SIZE, shuffle=True)      
net = Net().to(DEVICE)
criterion = torch.nn.CrossEntropyLoss()
optim = torch.optim.Adam(net.parameters(), lr=0.0001)

for epoch in range(100):
  cnt = 0
  ls = 0
  acc = 0
  bar = tqdm(dataloader)
  for batch, labels in bar:
    optim.zero_grad()
    x = net(batch)
    loss = criterion(x, labels)
    acc += torch.sum(torch.argmax(x, dim=1) == labels)
    cnt += batch.shape[0]
    ls += loss.item()
    bar.set_description(f"epoch={epoch}, loss={1000.0*ls/cnt:.3f}, acc={100.0*acc/cnt:.3f}%")
    loss.backward()

    optim.step()

    torch.save({
      "net": net.state_dict(),
      "optim": optim.state_dict()
    }, "model.pt")