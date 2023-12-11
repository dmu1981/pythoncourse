import torch

# Die Lernrate
eta = 0.001

# Unsere Gewichte
W = torch.tensor([1.0, 2.0, -3.0, -0.5], requires_grad=True)

# Unsere Daten
X = torch.tensor([
    [5.0, 6.0, 7.0, 8.0],
    [7.0, 6.0, 5.0, 4.0],
    [1.0, 3.0, 5.0, 7.0],
    [8.0, 5.0, 1.0, 4.0],
    ])

Y = torch.tensor([-1.0, 1.0, -1.0, 1.0])

# Das Skalarprodukt (Die Präditkion)
for epoch in range(1000):

    prediction = torch.functional.F.tanh(X @ W)
    #print(prediction)

    # Der Vorhersagefehler
    error = torch.mean((prediction - Y)**2)
    print(error)

    # Der Gradient
    error.backward()
    #print(W.grad)

    # Gradientenabstieg
    W = (W.data - eta * W.grad).requires_grad_(True)
    
print(prediction)    
print(W)


