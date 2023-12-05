import torch
from torch import nn

class Net(nn.Module):
  def __init__(self):
    super().__init__()
    self.sig = nn.Sigmoid()
    self.fc1 = nn.Linear(2, 2)
    self.fc2 = nn.Linear(2, 2)
    self.fc3 = nn.Linear(2, 2)

  def forward(self, x):
    x = self.sig(self.fc1(x))
    x = self.sig(self.fc2(x))
    x = self.sig(self.fc3(x))
    return x

net = Net()  
loss = nn.MSELoss()
optim = torch.optim.SGD(net.parameters(), lr=0.01)

optim.zero_grad()

X = torch.tensor([0.4, 0.5])
Y = torch.tensor([0.4, 1.0])

for epoch in range(1000):
  out = net(X)
  e = loss(out, Y)
  e.backward()

  if epoch % 100 == 0:
    print(e.item())
    
  optim.step()



