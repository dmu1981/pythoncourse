import numpy as np
import math
from matplotlib import pyplot as plt

X = np.array([[0,0],[0,1],[0,-1],[1,1],[-1,1],[-1,-1],[1,-1],[-1,0]])
K = np.zeros((8,8))
for i in range(8):
    for j in range(8):
        d = X[i,:] - X[j, :]
        #d = math.sqrt(d.T@d)
        d = d.T@d
        K[i,j] = math.exp(-d)

for row in range(K.shape[0]):
    rowstr = ' & '.join([f"{K[row][col]:.2f}" for col in range(K.shape[1])])
    print(f"{rowstr}\\\\")

#print("K:", K)
Kinv = np.linalg.inv(K)
p = Kinv @ np.array([-1,-1,-1,-1,1,1,1,1]).T
print('\\\\'.join([f"{p[idx]:.2f}" for idx in range(p.shape[0])]))
print("p:", p)

Q = np.array([0.75, -0.35])
k = np.zeros((8))
for j in range(8):
    d = Q - X[j, :]
    d = d.T@d
    k[j] = math.exp(-d)

print('\\\\'.join([f"{k[idx]:.2f}" for idx in range(k.shape[0])]))
print(k)
print(np.dot(p,k))

x_range = np.linspace(-1.5, 1.5, 20)
y_range = np.linspace(-1.5, 1.5, 20)
xx, yy = np.meshgrid(x_range, y_range)
z = np.zeros_like(xx)
for xidx, x in enumerate(x_range):
    for yidx, y in enumerate(y_range):
        k = np.zeros((8))
        for j in range(8):
            d = np.array([x,y]) - X[j, :]
            d = d.T@d
            k[j] = math.exp(-d)
        z[yidx, xidx] = np.dot(p, k)

plt.plot(X[4:, 0], X[4:, 1], "ok", ms=10)
plt.plot(X[4:, 0], X[4:, 1], "or", ms=8)

plt.plot(X[:4, 0], X[:4, 1], "ok", ms=10)
plt.plot(X[:4, 0], X[:4, 1], "ob", ms=8)

plt.plot([0.75], [-0.35], "ok", ms=10)
plt.plot([0.75], [-0.35], "og", ms=8)

plt.contourf(xx, yy, z, levels=25, cmap="jet")
plt.contour(xx, yy, z, levels=[0], colors=["k"])
plt.show()