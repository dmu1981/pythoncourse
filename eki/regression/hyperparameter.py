import numpy as np
from matplotlib import pyplot as plt

# Define our points
coords_x = np.linspace(-5,5,20)
coords_y = 0.5 + 2.8 * np.sin(0.7 * coords_x) - 0.3 * np.cos(0.7 * coords_x) 
coords_y += np.random.normal(0.0, 0.1, coords_y.size)

best_residuum = None
best_a1, best_a2 = None, None
best_model = None

# Do 2D hyperparamter search
for a1 in np.linspace(0.5, 1.5, 20):
  for a2 in np.linspace(0.5, 1.5, 20):
    # Our linear model is y = w_0 + w_1 * sin(a1 * x) + w_2 * cos(a2 * x) setup the X matrix and Y vector
    X = np.stack(
        [np.ones_like(coords_x), np.sin(a1 * coords_x), np.cos(a2 * coords_x)],
        axis=1
    )
    Y = coords_y

    # Estimate model paramaters using pseudo-inverse
    Xinv = np.linalg.inv(X.T @ X) @ X.T
    model = Xinv @ Y

    # Retrieve model parameters
    model = model.flatten()
    w0 = model[0]
    w1 = model[1]
    w2 = model[2]

    residuum = np.sum((w0 + w1 * np.sin(a1 * coords_x) + w2 * np.cos(a2 * coords_x) - coords_y)**2)
    if best_residuum is None or residuum < best_residuum:
      best_residuum = residuum
      best_model = model
      best_a1 = a1
      best_a2 = a2


w0 = best_model[0]
w1 = best_model[1]
w2 = best_model[2]
a1 = best_a1
a2 = best_a2
print(w0, w1, w2, a1, a2, best_residuum)

# Plot them
plt.plot(coords_x, coords_y, 'b*')
x = np.linspace(-5,5,150)
plt.plot(x, w0 + w1 * np.sin(a1 * x) + w2 * np.cos(a2 * x), "r")
plt.xlim((-5,5))
plt.ylim((-5,5))
plt.show()