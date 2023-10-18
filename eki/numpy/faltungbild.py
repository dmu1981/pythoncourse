from scipy import datasets, signal
import numpy as np

a = np.array([[1,4,3,3,0],[3,0,1,4,3],[2,3,3,2,2],[1,4,2,5,2],[5,4,0,1,1]])
b = np.array([[-3,1],[1,1]])

print(a)
print(b)
print(signal.convolve(a, b, mode="valid"))