import math

m, b, a, eta = 0, 9, 0, 0.01
x = [ 0.5, 1.0, 2.0, 3.0, 2.5]
y = [ 7.0, 5.0, 4.0, 3.0, 9.0]

for iter in range(10000):
    r = [m*x+b+a*x*x-y for (x,y) in zip(x,y)]

    grad_m, grad_b, grad_a, total_e = 0, 0, 0, 0
    for xi, ri in zip(x,r):
        total_e = total_e + math.log(math.cosh(ri))
        grad_a = grad_a + math.tanh(ri) * xi * xi
        grad_m = grad_m + math.tanh(ri) * xi
        grad_b = grad_b + math.tanh(ri)
    
    m, b, a  = m - eta * grad_m, b - eta * grad_b, a - eta * grad_a

print("total_e = {:3f}".format(total_e))
print("grad = ({:.3f},{:.3f},{:.3f})".format(grad_m, grad_b, grad_a))
print("m,b,a = {:.3f},{:.3f},{:.3f}".format(m, b, a))