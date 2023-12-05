import torch
from torch import nn

class Net(nn.Module):
  def __init__(self):
    super().__init__()
    self.sig = nn.Sigmoid()
    self.fc1 = nn.Linear(2, 4)
    self.fc2 = nn.Linear(4, 1)

  def forward(self, x):
    x = self.sig(self.fc1(x))
    x = self.sig(self.fc2(x))
    return x

net = Net()  
loss = nn.MSELoss()
optim = torch.optim.SGD(net.parameters(), lr=0.001)

optim.zero_grad()

X = torch.tensor([[0.0, 0.0], 
                  [1.0, 0.0], 
                  [0.0, 1.0], 
                  [1.0, 1.0]])

Y = torch.tensor([[0.0, 1.0, 1.0, 0.0]]).T

for epoch in range(10000):
  out = net(X)
  e = loss(out, Y)
  e.backward()
  
  if epoch % 1000 == 0:
    print(e.item())
    
  optim.step()



