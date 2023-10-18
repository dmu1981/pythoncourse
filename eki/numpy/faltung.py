from scipy import datasets, signal
import numpy as np

a = np.array([-2,  1, -1,  1])
b = np.array([ 2, -1, -1,  2])
c = np.array([ 1,  2, -2, -1])

print("Kommutativität")
print(signal.convolve(a, b))
print(signal.convolve(b, a))

print("Assoziativität")
print(signal.convolve(signal.convolve(a, b),c))
print(signal.convolve(a, signal.convolve(b, c)))

print("Distributivität")
print(signal.convolve(a, (b+c)))
print(signal.convolve(a, b) + signal.convolve(a, c))