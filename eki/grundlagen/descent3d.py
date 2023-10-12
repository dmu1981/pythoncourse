# Define our function F and its gradient
import numpy as np

def f(x):
    return (x[0]-3)**2 + (x[1]-2)**2 + (x[2]+5)**2

def gradf(x):
    return np.array([2*(x[0]-3), 2*(x[1]-2), 2*(x[2]+5)])   

# We can evaluate f by passing a coordinate to it
x0 = np.array([2.0,3.0,-3.0])
f(x0)

# We can also evaluate the gradient by passing a coordinate to it
gradf(x0)

# Now create a gradient descent scheme
x0 = np.array([2.0,3.0,-3.0])
eta = 0.3
for iter in range(3):
    print("Iteration ", iter)
    print(f"x: ({x0[0]:.3f}, {x0[1]:.3f}, {x0[2]:.3f})")
    print(f"f(x): {f(x0):.3f}")
    grad = gradf(x0)
    print(f"grad: ({grad[0]:.3f}, {grad[1]:.3f}, {grad[2]:.3f})")
    x0 = x0 - eta * grad
    print(" ")