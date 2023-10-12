import numpy as np
from matplotlib import pyplot as plt

def f(x):
    return x**4-10*x**2+10*x

def grad(x):
    return 4*x**3-20*x+10

x = np.linspace(-3,3,50)
y = [f(x) for x in x]

eta = 0.01

x0 = 0.6

x_arr, y_arr = [], []

for i in range(200):
    x_arr.append(x0)
    y_arr.append(f(x0))
    x0 = x0 - eta * grad(x0)
    #print(x0)


plt.plot(x,y, "r")
plt.plot(x_arr, y_arr, "b*")
plt.plot(x_arr, y_arr, "k", linewidth=0.5)
plt.show()