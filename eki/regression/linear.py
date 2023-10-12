import numpy as np
from matplotlib import pyplot as plt

# Define our points
coords_x = np.array([[1,2,3]]).T
coords_y = np.array([[3,4,4]]).T

# Our linear model is y = w_0 + w_1 * x, setup the X matrix and Y vector
X = np.concatenate(
    (np.ones_like(coords_x), coords_x),
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
print(w0, w1)

# Plot them
plt.plot(coords_x, coords_y, 'b*')
x = np.linspace(0,5,10)
plt.plot(x, w0 + w1 * x, "r")
plt.xlim((0,5))
plt.ylim((0,5))
plt.show()