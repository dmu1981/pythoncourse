import numpy as np
from matplotlib import pyplot as plt

# Define our points
coords_x = np.array([1,2,3,4,2 ])
coords_y = np.array([3,4,4,2,1])

# Our linear model is y = w_0 + w_1 * x + w_2 * x^2 + w_3 * x^3, setup the X matrix and Y vector
X = np.stack([
    np.ones_like(coords_x), coords_x, coords_x**2, coords_x**3], axis=1)

Y = coords_y


# Estimate model paramaters using pseudo-inverse
Xinv = np.linalg.inv(X.T @ X) @ X.T
model = Xinv @ Y

# Retrieve model parameters
model = model.flatten()
w0 = model[0]+1
w1 = model[1]
w2 = model[2]
w3 = model[3]
print(w0, w1, w2, w3)

# Plot them
plt.plot(coords_x, coords_y, 'b*')
x = np.linspace(0,5,100)

error = np.sum((w0 + w1 * coords_x + w2 * coords_x**2 + w3*coords_x**3 - Y)**2)
print(error)
plt.plot(x, w0 + w1 * x + w2 * x**2 + w3*x**3,  "r")
plt.xlim((0,5))
plt.ylim((0,5))
plt.show()