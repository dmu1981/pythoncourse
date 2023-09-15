import torch

# Tensoren haben ein shape und einen Datentyp
A = torch.rand(5,3)
print(A)
print(A.shape)
print(A.dtype)

# Man kann Tensoren mit gleichem Shape erzeugen über
B = torch.zeros_like(A)
print(B)