import torch

# Tensoren können direkt erzeugt werden
X = torch.tensor([[1,3],[4,2]])
print(X, end="\n\n")

# Tensoren können auch indirekt erzeugt werden
print(torch.zeros(3,3), end="\n\n")
print(torch.ones(3,3), end="\n\n")
print(torch.rand(3,3), end="\n\n")