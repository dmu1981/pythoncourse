import numpy as np
from matplotlib import pyplot as plt

# Define our points
coords_x = np.array([1,2,3,4,8])
coords_y = np.array([3,4,4,2,4])


# Our linear model is y = w_0 + w_1 * sin(a_1 * x) + w_2 * cos(x) setup the X matrix and Y vector
X = np.stack(
    [np.ones_like(coords_x), np.sin(1.7 * coords_x), np.cos(coords_x)],
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
x = np.linspace(-5,15,150)
plt.plot(x, w0 + w1 * np.sin(1.7*x) + w2 * np.cos(x), "r")
plt.xlim((-5,15))
plt.ylim((0,5))
plt.show()