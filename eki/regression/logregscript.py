import numpy as np
from matplotlib import pyplot as plt

x1_coords = np.array([1,3,4,6])
x2_coords = np.array([1,2,3,2])
X = np.stack(
  [np.ones_like(x1_coords), x1_coords, x2_coords], axis=1
)
y = np.array([0,0,1,1])

# Lineare regression
pseudoinv = np.linalg.inv(X.T@X)@X.T
model = pseudoinv@(2*y-1)
print("Linear model: ", model)
# Evaluate linear regression model
print("Linear prediction: ", X @ model)
P = 1.0 / (1.0 + np.exp(-X @ model))
print(P)

eta = 0.01
for _ in range(1000):
  P = 1.0 / (1.0 + np.exp(-X @ model))
  
  delta = (y-P)
  grad = np.array([
      np.sum(delta * np.ones_like(y)),
      np.sum(delta * x1_coords),
      np.sum(delta * x2_coords),
    ])
 
  model = model + eta * grad
  

plt.show()
print("Model: ", model)    