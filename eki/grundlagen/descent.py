import numpy as np

def f(p):
    return (p[0]-3)**2 + (p[1]-2)**2 + (p[2]+5)**2

def grad_f(p):
    return np.array([2*(p[0]-3), 2*(p[1]-2), 2*(p[2]+5)])

p = np.array([13, -5, 9])

eta = 0.05

for step in range(60):
    value = f(p)
    grad = grad_f(p)
    print(f"f({p[0]:5.2f}, {p[1]:5.2f}, {p[2]:5.2f}) = {value:7.2f}... grad = ({grad[0]:5.2f}, {grad[1]:5.2f}, {grad[2]:5.2f})")

    p = p - eta * grad