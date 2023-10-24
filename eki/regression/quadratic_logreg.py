import numpy as np
from matplotlib import pyplot as plt

# Define our points
coords_x1 = np.array([-2.5,  0, 0,   5,  0, 5, 5, 10, -5])
coords_x2 = np.array([-5, -5, 5, -10, 10, 0, 5, 5, 0])
class_y = np.array([-1,-1,-1,-1,1,1,1,1,1])

# Our linear model is y = w_0 + w1 * x1 + w_2 * x2 + w_3 * x1**2 + w_4 * x2**2 + w_5 * x1 * x2, setup the X matrix and Y vector
X = np.stack([np.ones_like(coords_x1), 
              coords_x1, 
              coords_x2, 
              coords_x1**2, 
              coords_x2**2, 
              coords_x1 * coords_x2], axis=1)
Y = class_y

# Estimate model paramaters using pseudo-inverse
Xinv = np.linalg.inv(X.T @ X) @ X.T
model = Xinv @ Y
print(model)

eta = 0.001

# Map class attributions to propabilities between 0 and 1 
Y = (Y + 1.0) / 2.0
# # Retrieve model parameters
model = model.flatten()

for epoch in range(6000):
    tildey = X @ model
    P = 1.0 / (1.0 +np.exp(-tildey))
    error = np.sum(Y*np.log(P)+(1-Y)*(np.log(1-P)))
    delta = Y - P
    #print("Delta: ", delta)

    grad = delta @ X
    #print(grad)
    # grad = np.array(
    #     [
    #       np.sum(delta * np.ones_like(coords_x1)),
    #       np.sum(delta * coords_x1),
    #       np.sum(delta * coords_x2),
    #       np.sum(delta * coords_x1**2),
    #       np.sum(delta * coords_x2**2),
    #       np.sum(delta * coords_x1*coords_x2),
    #     ]
    #     )
    # print(grad)
    mag = np.sqrt(grad.T @ grad)
    print(f"p(Data): {np.exp(error)*100.0:6.3}% ({error:.2f})     Gradient Magnitude: {mag:.2f}")

    model = model + eta * grad




print(grad)


w0 = model[0]
w1 = model[1]
w2 = model[2]
w3 = model[3]
w4 = model[4]
w5 = model[5]

print(f"w0: {w0:.4f}\nw1: {w1:.4f}\nw2: {w2:.4f}\nw3: {w3:.4f}\nw4: {w4:.4f}\nw5: {w5:.4f}")

# Plot them
blue_indices = (class_y == -1)
red_indices = (class_y == 1)
plt.plot(coords_x1[blue_indices], coords_x2[blue_indices], 'bo')
plt.plot(coords_x1[red_indices], coords_x2[red_indices], 'ro')

# Let us calculate the decision value for all points in the domain from -11 to 11
# and do a contour plot
x1, x2 = np.meshgrid(np.linspace(-11,11,50), np.linspace(-11,11,50))
z = w0 + w1 * x1 + w2 * x2 + w3 * x1**2 + w4 * x2**2 + w5 * x1 * x2
z = 1.0 / (1.0 + np.exp(-z))

plt.contourf(x1, x2, z, levels=np.linspace(0,1,29), alpha=.2, cmap="seismic")
plt.contour(x1, x2, z, levels=[0.5], colors=["k"])

plt.xlim((-11,11))
plt.ylim((-11,11))
plt.show()