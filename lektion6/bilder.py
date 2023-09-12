from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

CONTRAST = 0.5
BRIGHTNESS = -0.2

img = Image.open("lektion6/forest.jpeg")
img2 = np.array(img) / 255
img2 = 0.5 + (img2 - 0.5) * (1.0 + CONTRAST)
img2 = np.clip(img2 + BRIGHTNESS, 0.0, 1.0)
img2 = Image.fromarray(np.uint8(img2 * 255))

plt.subplot(1,2,1)
plt.imshow(img)
plt.subplot(1,2,2)
plt.imshow(img2)
plt.show()
