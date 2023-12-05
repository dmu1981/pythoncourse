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

N = 10000
for epoch in range(N):
    if W1.grad is not None:
        W1.grad.zero_()

    if W2.grad is not None:
        W2.grad.zero_()

    if W3.grad is not None:
        W3.grad.zero_()

    Z1 = W1 @ X
    O1 = torch.functional.F.sigmoid(Z1)
    if epoch == 0 or epoch == N-1:
        print("Z1: ", Z1)
        print("O1: ", O1)

    Z2 = W2 @ O1
    O2 = torch.functional.F.sigmoid(Z2)
    if epoch == 0 or epoch == N-1:
        print("Z2: ", Z2)
        print("O2: ", O2)

    Z3 = W3 @ O2
    O3 = torch.functional.F.sigmoid(Z3)
    if epoch == 0 or epoch == N-1:
        print("Z3: ", Z3)
        print("O3: ", O3)   

    E = torch.mean((O3 - Y)**2)
    if epoch % 1000 == 0:
        print(f"{E:.3f}")

    # Now calculate gradient
    
    E.backward()

    # Gradient descent
    W1 = (W1.data - eta * W1.grad).requires_grad_(True)
    W2 = (W2.data - eta * W2.grad).requires_grad_(True)
    W3 = (W3.data - eta * W3.grad).requires_grad_(True)

