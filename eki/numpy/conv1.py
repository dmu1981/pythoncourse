from scipy import datasets, signal
from matplotlib import pyplot as plt
import numpy as np

img = datasets.ascent()
result = signal.convolve(img, np.array([[1,-1],[-1,1]]))
result = np.clip(result, 0, 255)

fig, axs = plt.subplots(1,2)

axs[0].imshow(img, cmap="gray")
axs[1].imshow(result, cmap="gray")

plt.show()