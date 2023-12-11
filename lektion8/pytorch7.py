import torch

# Unsere Gewichte
W = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)

# Unsere Daten
X = torch.tensor([5.0, 6.0, 7.0, 8.0])

# Das Skalarprodukt
summe = torch.sum(torch.sin(W*X))
print(summe)

# Die Ableitung (nach W)
summe.backward()

# Der Gradient
print(W.grad)


