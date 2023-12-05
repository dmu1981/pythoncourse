import torch
from tqdm import tqdm 

# Die Lernrate
eta = 0.01

# Unsere Gewichte
W1 = torch.tensor([[ 2.5, -2.0], 
                   [-1.5,  1.5]], requires_grad=True)

W2 = torch.tensor([[ 3.5, -1.5], 
                   [-2.5,  2.5]], requires_grad=True)

W3 = torch.tensor([[ 4.5, -3.5], 
                   [-2.5,  3.5]], requires_grad=True)

# Unsere Daten
X = torch.tensor([0.4, 0.5])
Y = torch.tensor([0.4, 1.0])

Z1 = W1 @ X
O1 = torch.functional.F.sigmoid(Z1)
print("Z1: ", Z1)
print("O1: ", O1)

Z2 = W2 @ O1
O2 = torch.functional.F.sigmoid(Z2)
print("Z2: ", Z2)
print("O2: ", O2)

Z3 = W3 @ O2
O3 = torch.functional.F.sigmoid(Z3)
print("Z3: ", Z3)
print("O3: ", O3)   

E = torch.mean((O3 - Y)**2)
print(f"{E:.3f}")

    