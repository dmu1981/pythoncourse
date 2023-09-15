import torch

# Ähnlich wie bei NumPy können viele Operationen auf
# Matrizen direkt ausgedrückt werden
A = torch.tensor([[2.0,1.0], [3.0,4.0]])
B = torch.tensor([[1.0,4.0], [5.0,2.0]])
print("A+B")
print(A+B, end="\n\n")

print("A*B")
print(A*B, end="\n\n")

print("A@B")
print(A@B, end="\n\n")

# Für andere Operationen gibt es spezielle Funktionen
print("inverse")
C = torch.inverse(A)
print(C, end="\n\n")
