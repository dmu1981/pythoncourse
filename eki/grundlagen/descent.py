import numpy as np
from matplotlib import pyplot as plt

# Define our function and its gradient
def f(x):
    return x**4-10*x**2+10*x

def grad(x):
    return 4*x**3-20*x+10

# Plot the function itself
x = np.linspace(-4.0, 4.0, 50)
plt.plot(x, [f(x) for x in x], "r")

# Do a gradient descent and plot the respective positions as well
x0 = 0.8
eta = 0.01
xcoords, ycoords = [], []
for i in range(40):
    # Remember coordinate for plotting
    xcoords.append(x0)
    ycoords.append(f(x0))
    
    # Gradient descent
    x0 = x0 - eta * grad(x0)    

x = np.linspace(-4.0, 4.0, 50)
plt.plot(x, [f(x) for x in x], "r")

plt.plot(xcoords, ycoords, "k", linewidth=0.5)
plt.plot(xcoords, ycoords, "b*")