import numpy as np

x = np.linspace(-2,2,9)
print("x\n",x)

y = np.linspace(-3,3,9)
print("y\n",y)

xx,yy = np.meshgrid(x,y)
print("xx\n",xx)
print("yy\n",yy)