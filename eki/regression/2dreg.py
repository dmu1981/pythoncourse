import numpy as np
from matplotlib import pyplot as plt

# Define our points
coords_x1 = np.array([1,2,3,2,1,4,3,2,3,2,1])
coords_x2 = np.array([3,2,3,1,1,1,5,3,2,2,3])
coords_y = np.array([3,4,5,2,3,5,1,3,3,2,1])

# Our linear model is y = w_0 + w_1 * x_1 + w_2 * x_2, setup the X matrix and Y vector
X = np.stack([
    np.ones_like(coords_x1), coords_x1, coords_x2], axis=1)
Y = coords_y
print(X)

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
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(coords_x1, coords_x2, coords_y, color='b')

x1, x2 = np.meshgrid(np.linspace(0,5,10), np.linspace(0,5,10))
z = w0 + w1 * x1 + w2 * x2 
ax.plot_surface(x1,x2,z, alpha=0.1, color="red")

# x = np.linspace(0,5,10)
# plt.plot(x, w0 + w1 * x, "r")
# plt.xlim((0,5))
# plt.ylim((0,5))
plt.show()