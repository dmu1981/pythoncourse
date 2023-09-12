import numpy as np
from matplotlib import pyplot as plt

x = np.linspace(-3.0, 3.0, 9)
print(x)
y = x**2
print(y)
plt.plot(x,y)
plt.show()