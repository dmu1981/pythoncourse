import numpy as np
from matplotlib import pyplot as plt

x1_coords = np.array([1,3,4,6])
x2_coords = np.array([1,2,3,2])
X = np.stack(
  [np.ones_like(x1_coords), x1_coords, x2_coords], axis=1
)
y = np.array([0,0,1,1])

# Lineare regression
pseudoinv = np.linalg.inv(X.T@X)@X.T
model = pseudoinv@(2*y-1)
print("Linear model: ", model)
# Evaluate linear regression model
print("Linear prediction: ", X @ model)
P = 1.0 / (1.0 + np.exp(-X @ model))
print(P)

N_STEPS = 720
STEPS_TO_SHOW = 24
SHOW_EVERY = N_STEPS // STEPS_TO_SHOW 
rows = 4
cols = 6
fig, axs = plt.subplots(rows, cols)

step = 0
eta = 0.1
cnt = 0
while step < N_STEPS:
  # Evaluate logistic regression model
  P = 1.0 / (1.0 + np.exp(-X @ model))
  
  delta = (y-P)
  print("delta * ones: ", delta)
  print("delta * x1: ", delta * x1_coords)
  print("delta * x2: ", delta * x2_coords)
  grad = np.array(
    [
      np.sum(delta * np.ones_like(y)),
      np.sum(delta * x1_coords),
      np.sum(delta * x2_coords),
    ]
    )
  print("grad: ", grad)
  
  if step % SHOW_EVERY == 0:
    print("Step: ", step)
    print("Logistic propabilities: ", P)
    error = np.sum(y*np.log(P)+(1-y)*(np.log(1-P)))
    print("Error:", error)

    x1, x2 = np.meshgrid(np.linspace(0,7,30), np.linspace(0,4,30))
    z = model[0] + model[1] * x1 + model[2] * x2
    z = 1.0 / (1.0 + np.exp(-z))

    print(cnt)
    a = axs[cnt//cols, cnt%cols]
    
    a.plot(x1_coords[y == 1], x2_coords[y == 1], 'r*')
    a.plot(x1_coords[y == 0], x2_coords[y == 0], 'b*')
    #a.contourf(x1, x2, z, levels=[-100,0.5, 100], colors=["b","r"], alpha=.2)
    a.contourf(x1, x2, z, levels=np.linspace(0,1,19), alpha=.5, cmap="seismic")
    a.contour(x1, x2, z, levels=[0.5], colors=["k"])
    a.set_title(f"Step: {step}, Error: {error:.3f}")
    
    cnt = cnt + 1
    print("Gradient:", grad)
    

  #print("Gradient:", grad)
  #print(" ")
  model = model + eta * grad
  #print(model)
  step = step + 1 
  

plt.show()
print("Model: ", model)    