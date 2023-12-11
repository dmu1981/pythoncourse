import torch
from torch import nn

eta = 0.01

sig = nn.Sigmoid()
fc1 = nn.Linear(2, 4)
fc2 = nn.Linear(4, 4)
fc3 = nn.Linear(4, 1)

loss = nn.MSELoss()


X = torch.tensor([0.4, 0.5])
Y = torch.tensor([0.4])

for epoch in range(10000):
  x = X
  x = sig(fc1(x))
  x = sig(fc2(x))
  x = sig(fc3(x))
  e = loss(x, Y)
  e.backward()
  if epoch % 1000 == 0:
    print(e.item())
    
  fc1.weight.data = (fc1.weight.data - eta * fc1.weight.grad).requires_grad_(True)
  fc2.weight.data = (fc2.weight.data - eta * fc2.weight.grad).requires_grad_(True)
  fc3.weight.data = (fc3.weight.data - eta * fc3.weight.grad).requires_grad_(True)
  



