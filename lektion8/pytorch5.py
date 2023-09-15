import torch

A = torch.rand(2,2).to("cuda")
B = torch.rand(2,2).to("cuda")
C = torch.rand(2,2)

print(A+B)
print(A+C)
