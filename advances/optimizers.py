import numpy as np
from matplotlib import pyplot as plt 

# Generate some data
x = np.linspace(0, 10, 1000)
y = 0.0 + 0.0 * x + - 0.0 * x**2 + 0.5 * np.sin(0.07 * x * 2 * 3.14)

for idx, xv in enumerate(x):
  y[idx] += 0.1 * np.random.normal(0.0, 1.0, size=1)

# Momentum
momentum = 0
grad_momentum = []
for grad in y:
  momentum = 0.8 * momentum + grad
  grad_momentum.append(momentum)

# AdaGrad
sum_of_squares = [0]
grad_adagrad = []
for grad in y:
  sum_of_squares.append(sum_of_squares[-1] + grad**2)
  grad_adagrad.append(grad / np.sqrt(sum_of_squares[-1]))

# AdaDelta
alpha = 0.9
sum_of_squares_adadelta = [0]
grad_adadelta = []
for grad in y:
  sum_of_squares_adadelta.append(alpha * sum_of_squares_adadelta[-1] + (1-alpha)*grad**2)
  grad_adadelta.append(grad / np.sqrt(sum_of_squares_adadelta[-1]))

# Adam
beta1 = 0.9
beta2 = 0.999
sum_of_grad_adam = [0]
sum_of_squares_adam = [0]
grad_adam = []
m = 0
v = 0
b1cor = 1
b2cor = 1
for grad in y:
  m = beta1 * m + (1 - beta1) * grad
  v = beta2 * v + (1 - beta2) * (grad ** 2)
  b1cor *= beta1
  b2cor *= beta2
  sum_of_grad_adam.append(m / (1 - b1cor))
  sum_of_squares_adam.append(v / (1 - b2cor))

  grad_adam.append(sum_of_grad_adam[-1] / np.sqrt(sum_of_squares_adam[-1]))

#print(grad_adam)
#exit()
 
fig, axs = plt.subplots(1,5)

axs[0].plot(x,y, 'r')
axs[0].plot(x,y, '.')
axs[0].set_title("SGD")


axs[1].plot(x,grad_momentum, 'r')
axs[1].plot(x,y, '.')
axs[1].set_title("SGD with Momentum")


#axs[2].plot(x,np.sqrt(sum_of_squares[1:]), 'g')
axs[2].plot(x,grad_adagrad, 'r')
axs[2].plot(x,y, '.')
axs[2].set_title("AdaGrad")


#axs[3].plot(x,np.sqrt(sum_of_squares_adadelta[1:]), 'g')
axs[3].plot(x,grad_adadelta, 'r')
axs[3].plot(x,y, '.')
axs[3].set_title("AdaDelta")


#axs[4].plot(x,np.sqrt(sum_of_squares_adam[1:]), 'g')
axs[4].plot(x,y, '.')
axs[4].plot(x,grad_adam, 'r')
axs[4].set_title("Adam")

plt.show()
