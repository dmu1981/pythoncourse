import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

gaussians = 200

centers = []
covariances = []
norm =  []
for _ in range(gaussians):
  center = np.random.uniform(-10,10,size=(2,1))
  v1 = np.random.uniform(0.2,2,1)[0]
  v2 = np.random.uniform(0.2,2,1)[0]
  m = np.minimum(v1,v2) * 0.95
  cov = np.random.uniform(-m,m,1)[0]
  
  norm.append(0.5*np.random.uniform(0.2, 2))
  centers.append(center)
  cov = np.array([[v1, cov],[cov, v2]])
  covariance = np.linalg.inv(cov)
  covariances.append(covariance)

xrange = np.linspace(-10, 10, 60)
yrange = np.linspace(-10, 10, 60)  

xx, yy = np.meshgrid(xrange, yrange)
z = np.zeros_like(xx)
for yidx, y in enumerate(yrange):
  for xidx, x in enumerate(xrange):
    v = np.array([[x],[y]])
    for idx in range(gaussians):
      d = ((v - centers[idx]).T @ covariances[idx] @ (v - centers[idx]))[0][0]
      z[xidx][yidx] += norm[idx] * np.exp(-0.5 * d)

    z[xidx][yidx] += 0.06*(x**2 + y**2)

z = z / np.max(z)

eta = 0.1

def evaluate_z(x,y):
  xidx = np.argmin(np.abs(x - xrange))
  yidx = np.argmin(np.abs(y - yrange))
  z[xidx][yidx]

def evaluate_grad(x):
  xidx = np.argmin(np.abs(x[0] - xrange))
  yidx = np.argmin(np.abs(x[1] - yrange))
  dx = xrange[xidx+1] - xrange[xidx]
  dy = yrange[yidx+1] - xrange[yidx]
  z0=z[xidx][yidx]
  zx=z[xidx+1][yidx]
  zy=z[xidx][yidx+1]
  return np.array([
    (zx-z0)/dx,
    (zy-z0)/dy
    ])

class Optimizer:
  def __init__(self):
    self.trace = None
    self.last_pos = None

  def update(self, pos):
      if self.trace is None:
        self.trace = pos.reshape(2,1)
        self.last_pos = pos.reshape(2,1)
        return True
      else:
        self.trace = np.concatenate((self.trace, pos.reshape(2,1)),axis=1)
        delta = pos.reshape(2,1) - self.last_pos
        
        mag = np.sqrt(delta.T @ delta)
        if mag < 0.001 and self.trace.shape[1] > 20:
          print(f"Convergence after {self.trace.shape[1]} steps")
          return False
        
        self.last_pos = pos.reshape(2,1)
        
        return True

class SGD(Optimizer):
  def __init__(self):
    super().__init__()

  def run(self, pos):
    for i in range(1500):
      if not self.update(pos):
        break

      grad = evaluate_grad(pos)
      pos = pos - 10 * eta * grad

class Momentum(Optimizer):
  def __init__(self):
    super().__init__()
    self.moment = np.array([0,0])

  def run(self, pos):
    for i in range(1500):
      if not self.update(pos):
        break

      self.moment = 0.9 * self.moment + evaluate_grad(pos)      
      pos = pos - eta * self.moment

class AdaGrad(Optimizer):
  def __init__(self):
    super().__init__()
    self.grad2 = np.array([0.0,0.0])

  def run(self, pos):
    for i in range(1500):
      if not self.update(pos):
        break

      grad = evaluate_grad(pos)
      self.grad2 += (grad ** 2)
      pos = pos - eta * grad / np.sqrt(self.grad2) 

class AdaDelta(Optimizer):
  def __init__(self):
    super().__init__()
    self.grad2 = np.array([0.0,0.0])

  def run(self, pos):
    for i in range(1500):
      if not self.update(pos):
        break

      grad = evaluate_grad(pos)
      self.grad2 = self.grad2 * 0.9 + 0.1 * (grad ** 2)
      pos = pos - eta * grad / np.sqrt(self.grad2) 

class Adam(Optimizer):
  def __init__(self):
    super().__init__()
    self.grad1 = np.array([0.0,0.0])
    self.grad2 = np.array([0.0,0.0])
    self.beta1 = 0.9
    self.beta2 = 0.999
    self.beta1cor = 1.0
    self.beta2cor = 1.0

  def run(self, pos):
    for i in range(1500):
      if not self.update(pos):
        break

      self.beta1cor *= self.beta1
      self.beta2cor *= self.beta2

      grad = evaluate_grad(pos)
      self.grad1 = self.grad1 * self.beta1 + (1-self.beta1) * (grad)
      self.grad2 = self.grad2 * self.beta2 + (1-self.beta2) * (grad ** 2)
      pos = pos - eta * (self.grad1 / (1.0-self.beta1cor)) / np.sqrt(self.grad2 / (1.0-self.beta2cor)) 

pos = np.array([8.0,8.0])

optimSGD = SGD()
optimSGD.run(pos)

optimMomentum = Momentum()
optimMomentum.run(pos)

optimAdagrad = AdaGrad()
optimAdagrad.run(pos)

optimAdaDelta = AdaDelta()
optimAdaDelta.run(pos)

optimAdam = Adam()
optimAdam.run(pos)

optims = [(optimSGD, "k"), 
          (optimMomentum, "g"),
          (optimAdagrad, "b"),  
          (optimAdaDelta, "c"),
          (optimAdam, "r")]

maxSteps = 999999
for optim, color in optims:
  if optim.trace.shape[1] < maxSteps:
    maxSteps = optim.trace.shape[1]

plt.ion()

plt.contourf(xx,yy,z,levels=np.linspace(0,1,150), cmap="bone")


delay = 20 / optim.trace.shape[1]
#plt.plot(optim.trace[0],optim.trace[1],"k", linewidth=0.1)
plt.pause(3)
for upto in tqdm(range(2,maxSteps,1)):
  for optim, color in optims:
    plt.plot(optim.trace[0][(upto-2):upto],optim.trace[1][(upto-2):upto],color,linewidth=2)

  plt.pause(delay)

print("Done")
plt.pause(100)
plt.show()