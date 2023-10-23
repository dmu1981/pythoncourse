import numpy as np
from matplotlib import pyplot as plt

#print(np.linspace(-20,20,10))
#exit()
# The data
x1_coords = np.array([2,1,3,2,3,3,2,3,5,5,6,4,7,6,6,6,7,7])
x2_coords = np.array([3,3,1,2,3,2,4,5,3,4,5,4,7,7,6,6,5,4])
X = np.stack(
  [np.ones_like(x1_coords), x1_coords, x2_coords], axis=1
)
y = np.array([1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0])

# Lineare regression
pseudoinv = np.linalg.inv(X.T@X)@X.T
model = pseudoinv@(2*y-1)
print("Linear model: ", model)
# Evaluate linear regression model
print("Linear prediction: ", X @ model)

N_STEPS = 720
STEPS_TO_SHOW = 24
SHOW_EVERY = N_STEPS // STEPS_TO_SHOW 
rows = 4
cols = 6
fig, axs = plt.subplots(rows, cols)

step = 0
eta = 0.01
cnt = 0
while step < N_STEPS:
  # Evaluate logistic regression model
  P = 1.0 / (1.0 + np.exp(-X @ model))
  
  delta = (y-P)
  grad = np.array(
    [
      np.sum(delta * np.ones_like(y)),
      np.sum(delta * x1_coords),
      np.sum(delta * x2_coords),
    ]
    )
  
  if step % SHOW_EVERY == 0:
    print("Step: ", step)
    print("Logistic propabilities: ", P)
    error = np.sum(y*np.log(P)+(1-y)*(np.log(1-P)))
    print("Error:", error)

    x1, x2 = np.meshgrid(np.linspace(0,8,30), np.linspace(0,8,30))
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

  #print("Gradient:", grad)
  #print(" ")
  model = model + eta * grad
  step = step + 1 

plt.show()
print("Model: ", model)    