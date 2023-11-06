import numpy as np
from matplotlib import pyplot as plt

# Define our points
N = 200
coords_x1 = np.random.uniform(-1, 1, N)
coords_x2 = np.random.uniform(-1, 1, N)
class_y = (coords_x1**2 + coords_x2**2 + np.random.normal(0.0, 0.4, N)) < 0.5
class_y = 2*class_y - 1

# Our linear model is y = w_0 + w1 * x1 + w_2 * x2 + w_3 * x1**2 + w_4 * x2**2 + w_5 * x1 * x2, setup the X matrix and Y vector
X = np.stack([np.ones_like(coords_x1), 
              coords_x1, 
              coords_x2, 
              coords_x1**2, 
              coords_x2**2, 
              coords_x1**3,
              coords_x2**3,
              coords_x1*coords_x2,
              coords_x1*(coords_x2**2),
              (coords_x1**2)*coords_x2,
              np.abs(coords_x1)**0.5,
              np.abs(coords_x2)**0.5,
              ], axis=1)

#plt.ion()
for l in np.linspace(0, 10, 100):
    #plt.figure()
    plt.clf()
    Xinv = np.linalg.inv(X.T @ X + l * np.eye(12)) @ X.T
    model = Xinv @ class_y
    #print(model)
    #print(np.sum(model**2))
    model = model.flatten()
    w0 = model[0]
    w1 = model[1]
    w2 = model[2]
    w3 = model[3]
    w4 = model[4]
    w5 = model[5]
    w6 = model[6]
    w7 = model[7]
    w8 = model[8]
    w9 = model[9]
    w10 = model[10]
    w11 = model[11]


    # Plot them
    blue_indices = (class_y == -1)
    red_indices = (class_y == 1)
    plt.plot(coords_x1[blue_indices], coords_x2[blue_indices], 'bo')
    plt.plot(coords_x1[red_indices], coords_x2[red_indices], 'ro')

    # Let us calculate the decision value for all points in the domain from -11 to 11
    # and do a contour plot
    x1, x2 = np.meshgrid(np.linspace(-11,11,150), np.linspace(-11,11,150))
    z = w0 + w1 * x1 + w2 * x2 + w3 * x1**2 + w4 * x2**2 + w5 * x1**3 + w6 * x2**3 + w7 * x1 * x2 + w8 * x1 * (x2**2) + w9 * (x1**2) * x2 + w10 * (np.abs(x1)**0.5) + w11 * (np.abs(x2)**0.5)
    plt.contourf(x1, x2, z, levels=[-10,0,10], alpha=.2, cmap="seismic")
    plt.contour(x1, x2, z, levels=[0], colors=["k"])

    error = np.sum((X @ model - class_y)**2)
    print(error)
    plt.title(f"||w|| = {np.sum(model**2):.3f}, error = {error:.3f}")
    plt.xlim((-1,1))
    plt.ylim((-1,1))
    plt.pause(0.1)
    #plt.show()