import numpy as np
from matplotlib import pyplot as plt
u1_range = np.linspace(0,1, 20)
u2_range = np.linspace(0,0.9999, 20)

u1, u2 = np.meshgrid(u1_range, u2_range)
print(u1)

theta = 2 * np.pi * u1
R = np.sqrt(-2.0 * np.log(1 - u2))

Y1 = R * np.cos(theta)
Y2 = R * np.sin(theta)

print(Y1)
print(Y2)

plt.scatter(Y1, Y2, c=u2, cmap="seismic")
plt.show()