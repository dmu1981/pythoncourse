import numpy as np
from matplotlib import pyplot as plt

# Create a linear space between 0 and 5
x    = np.linspace(-3,3,100)

fig, axs = plt.subplots(2,3)
axs[0,0].plot(x,np.sinh(x), "r")
axs[0,0].set_title("sinh")
axs[1,0].plot(x,np.cosh(x), "r")

axs[0,1].plot(x,np.cosh(x), "r")
axs[0,1].set_title("cosh")
axs[1,1].plot(x,np.sinh(x), "r")

axs[0,2].plot(x,np.tanh(x), "r")
axs[0,2].set_title("tanh")
axs[1,2].plot(x,1-np.tanh(x)**2, "r")

plt.show()