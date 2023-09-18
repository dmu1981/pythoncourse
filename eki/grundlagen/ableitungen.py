import numpy as np
from matplotlib import pyplot as plt

# Create a linear space between 0 and 5
x    = np.linspace(0,5,100)

# Calculate y-values
y    = x * np.exp(-x)

# Calculate derivative
dydx = np.exp(-x) - x * np.exp(-x) 

# Approximate derivative numerically
n = (y[1:] - y[:-1]) / (x[1:] - x[:-1])

# Plot all
plt.plot(x,y, "b")
plt.plot(x,dydx, "g")
plt.plot(x[:-1],n, "r")
plt.plot(x,np.zeros_like(x),"k--")
plt.show()