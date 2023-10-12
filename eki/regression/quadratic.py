import numpy as np
from matplotlib import pyplot as plt

# Define our points
coords_x = np.array([1,2,3,4]).T
coords_y = np.array([3,4,4,2]).T

# Our linear model is y = w_0 + w_1 * x, w_2 * x**2 setup the X matrix and Y vector
X = np.stack(
    [np.ones_like(coords_x), coords_x, coords_x**2],
    axis=1
)
Y = coords_y

# Estimate model paramaters using pseudo-inverse
Xinv = np.linalg.inv(X.T @ X) @ X.T
model = Xinv @ Y
print(model)

# Retrieve model parameters
model = model.flatten()
w0 = model[0]
w1 = model[1]
w2 = model[2]
print(w0, w1, w2)

# Plot them
plt.plot(coords_x, coords_y, 'b*')
x = np.linspace(0,5,50)
plt.plot(x, w0 + w1 * x + w2 * x**2, "r")
plt.xlim((0,5))
plt.ylim((0,5))
plt.show()