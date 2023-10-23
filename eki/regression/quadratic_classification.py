import numpy as np
from matplotlib import pyplot as plt

# Define our points
coords_x1 = np.array([-5,  0, 0,   5,  0, 5, 5, 10, -5])
coords_x2 = np.array([-5, -5, 0, -10, 10, 0, 5, 5, 0])
class_y = np.array([-1,-1,-1,-1,1,1,1,1,1])

# Our linear model is y = w_0 + w1 * x1 + w_2 * x2 + w_3 * x1**2 + w_4 * x2**2, setup the X matrix and Y vector
X = np.stack([np.ones_like(coords_x1), coords_x1, coords_x2, coords_x1**2, coords_x2**2], axis=1)
Y = class_y
#print(X)

# Estimate model paramaters using pseudo-inverse
Xinv = np.linalg.inv(X.T @ X) @ X.T
model = Xinv @ Y
#print(model)

# # Retrieve model parameters
model = model.flatten()
w0 = model[0]
w1 = model[1]
w2 = model[2]
w3 = model[3]
w4 = model[4]
print(f"w0: {w0:.4f}\nw1: {w1:.4f}\nw2: {w2:.4f}\nw3: {w3:.4f}\nw4: {w4:.4f}")

# Plot them
blue_indices = (class_y == -1)
red_indices = (class_y == 1)
plt.plot(coords_x1[blue_indices], coords_x2[blue_indices], 'bo')
plt.plot(coords_x1[red_indices], coords_x2[red_indices], 'ro')

# Let us calculate the decision value for all points in the domain from -11 to 11
# and do a contour plot
x1, x2 = np.meshgrid(np.linspace(-11,11,30), np.linspace(-11,11,30))
z = w0 + w1 * x1 + w2 * x2 + w3 * x1**2 + w4 * x2**2 
plt.contourf(x1, x2, z, levels=[-10,0, 10], colors=["b","r"], alpha=.2)
plt.contour(x1, x2, z, levels=[0], colors=["k"])

plt.xlim((-11,11))
plt.ylim((-11,11))
plt.show()