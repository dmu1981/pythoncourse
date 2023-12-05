import numpy as np
from matplotlib import pyplot as plt 
from tqdm import tqdm

N_OUTLIERS = 1
X = np.linspace(1, 9, 30)
Y = 7 - 0.7 * X + np.random.normal(0.0, 1.5, size=X.shape)

X = np.concatenate((X, np.linspace(1, 9, N_OUTLIERS)))
Y = np.concatenate((Y, np.random.normal(6, 4, size=(N_OUTLIERS))))

omega = np.array([-0.4, 6])
omega_logcosh = omega
omega_mse = omega

plt.ion()
x = np.linspace(0,10,100)
bar = tqdm(range(300))
for _ in bar:
  plt.clf()
  plt.plot(X[:-N_OUTLIERS], Y[:-N_OUTLIERS], 'gD')
  plt.plot(X[-N_OUTLIERS:], Y[-N_OUTLIERS:], 'rD')
  
  y = omega_logcosh[0] * x + omega_logcosh[1]
  plt.plot(x, y, "m")
  y = omega_mse[0] * x + omega_mse[1]
  plt.plot(x, y, "k", linewidth=0.2)

  plt.xlim(0, 10)
  plt.ylim(0, 10)
  plt.pause(0.01)

  for _ in range(20):
    residuum_logcosh = X * omega_logcosh[0] + omega_logcosh[1] - Y
    error_logcosh = np.sum(np.log(np.cosh(residuum_logcosh)))
    grad_logcosh = np.sum(np.tanh(residuum_logcosh) * np.stack((X, np.ones(X.shape[0]))), axis=1)
    omega_logcosh = omega_logcosh - 0.001 * grad_logcosh

  residuum_mse = X * omega_mse[0] + omega_mse[1] - Y
  error_logcosh = np.sum(residuum_logcosh**2)
  grad_mse = np.sum((residuum_mse) * np.stack((X, np.ones(X.shape[0]))), axis=1)
  omega_mse = omega_mse - 0.0001 * grad_mse
  
  bar.set_description(f"error={error_logcosh:.2f}, ||w||={np.sqrt(np.sum(grad_logcosh**2)):.3f}, m={omega_logcosh[0]:.2f}, b={omega_logcosh[1]:.2f}")