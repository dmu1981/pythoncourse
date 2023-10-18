from scipy import datasets, signal
from matplotlib import pyplot as plt
import numpy as np

img = datasets.face()

red = img[:,:,0]
green = img[:,:,1]
blue = img[:,:,2]

mask_red = np.array([
    [0,-1,0],
    [-1,-2,-1],
    [0,-1,0]
    ])

mask_green = np.array([
    [0,2,0],
    [2,4,2],
    [0,2,0]
    ])

mask_blue = np.array([
    [0,-1,0],
    [-1,-2,-1],
    [0,-1,0]
    ])

conv_red = signal.convolve(red, mask_red)
conv_green = signal.convolve(green, mask_green)
conv_blue = signal.convolve(blue, mask_blue)

fig, axs = plt.subplots(4,3)

axs[0,0].axis("off")
axs[0,1].imshow(img)
axs[0,2].axis("off")

red = red.reshape(red.shape[0], red.shape[1], 1)
axs[1,0].imshow(np.concatenate([red, np.zeros_like(red), np.zeros_like(red)], axis=2))

green = green.reshape(green.shape[0], green.shape[1], 1)
axs[1,1].imshow(np.concatenate([np.zeros_like(green), green, np.zeros_like(green)], axis=2))

blue = blue.reshape(blue.shape[0], blue.shape[1], 1)
axs[1,2].imshow(np.concatenate([np.zeros_like(blue), np.zeros_like(blue), blue], axis=2))

axs[2,0].imshow(conv_red, cmap="gray")
axs[2,1].imshow(conv_green, cmap="gray")
axs[2,2].imshow(conv_blue, cmap="gray")

res = conv_red+conv_green+conv_blue
res = np.maximum(0, res)
axs[3,0].axis("off")
axs[3,1].imshow(res, cmap="gray")
axs[3,2].axis("off")

plt.show()

