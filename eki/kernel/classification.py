import numpy as np
from matplotlib import pyplot as plt

def kernel(x1, x2):
  return np.exp(-0.5*np.sum((x1-x2)**2))
  return np.power(0.9*np.dot(x1, x2)+0.2,6)
  #return np.tanh(np.dot(x1,x2)/18+1)

N = 50
coords_red  = np.concatenate([
  np.array([[2,0],[0,1]]) @ np.random.normal(0, 1, (2,N)) + np.array([[0,3]]).T,
  np.array([[1,0],[0,2]]) @ np.random.normal(0, 1, (2,N)) + np.array([[0,-2]]).T,
],axis=1)

coords_blue  = np.concatenate([
  np.array([[1,-0.5],[0,2]]) @ np.random.normal(0, 1, (2,N)) + np.array([[-3,0]]).T,
  np.array([[1,0.5],[0,2]]) @ np.random.normal(0, 1, (2,N)) + np.array([[3,0]]).T
], axis=1)

X = np.concatenate([coords_red, coords_blue], axis=1)
Y = np.concatenate([-np.ones(coords_red.shape[1]), np.ones(coords_blue.shape[1])])

# Kernel Matrix aufstellen
K = np.zeros((X.shape[1], X.shape[1]))
for index1, x1 in enumerate(X.T):
  for index2, x2 in enumerate(X.T):
    K[index1][index2] = kernel(x1, x2)

# Kernel Matrix invertieren
K = np.linalg.inv(K + 0.01*np.eye(4*N))

# Modelparameter im Feature-Space berechnen
model = K @ Y

# Draw decision boundary
x_range = np.linspace(-6,6,25)
y_range = np.linspace(-6,6,25)
xx, yy = np.meshgrid(x_range, y_range)
z = np.zeros_like(xx)

for indexx, x in enumerate(x_range):
  for indexy, y in enumerate(y_range):
    # Make this coordinate a point so we can call the kernel function
    pt1 = np.array([x,y])

    # Calculate kernel for this point 
    vec = np.zeros_like(model)
    for idx, pt in enumerate(X.T):  
      vec[idx] = kernel(pt1, pt)

    # Calculate model prediction
    z[indexx, indexy] = model @ vec



plt.plot(coords_red[0], coords_red[1], "r.")
plt.plot(coords_blue[0], coords_blue[1], "b.")

plt.contourf(xx,yy,z,levels=[-100,0,100], cmap="seismic", alpha=0.2)
plt.contour(xx,yy,z,levels=[0], colors="k", linewidths=.5)

plt.xlim([-6,6])
plt.ylim([-6,6])
plt.show()