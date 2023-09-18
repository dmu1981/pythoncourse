import numpy as np
from matplotlib import pyplot as plt

# Create linear spaces in X and Y direction
x = np.linspace(-5, 5, 100)
y = np.linspace(-3, 3, 100)

# Turn into a meshgrid for evaluation of z
xv, yv = np.meshgrid(x,y)

# Evaluate Z (height)
z = xv ** 2  - yv ** 3

# Plot in 3D
X = -2.5
Y = -2.5
XS, YS, ZS = [], [], []

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(xv, yv, z, alpha=0.6)

# Do 50 steps
for i in range(50):  
  # Calculate Z coordinate
  Z = X ** 2 - Y ** 3

  # Append current coordianates to our list for plotting
  XS.append(X)
  YS.append(Y)
  ZS.append(Z)
  
  # Calculate gradient 
  DZDX =   2 * X
  DZDY = - 3 * (Y**2)

  # Normalize
  v = np.array([DZDX, DZDY])
  v = v / (0.1 + np.sum(v**2)) * 0.45

  # Step downwards
  X -= v[0]
  Y -= v[1]

# Do a scatter plot
ax.scatter(XS, YS, ZS, s=50, c="red")

# Label the axis
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")

# And show
plt.show()