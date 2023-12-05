import torch
from torch import nn

class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.input_dim = 12
        self.hidden_dim = 64
        self.out_dim = 5

        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, batch_first = True, num_layers=1)
        self.linear = nn.Linear(self.hidden_dim, self.out_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        print(out.shape4)
        x = self.linear(out[:, -1, :])
        return x

net = MyModule()

x = torch.randn((32, 100, 12))
out = net(x)
print(out.shape)
