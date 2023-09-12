from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider, Button

CONTRAST = 0.0
BRIGHTNESS = 0.0

img = Image.open("lektion6/forest.jpeg")
fig, ax = plt.subplots(1,2)
ax[0].imshow(img)
ax[1].imshow(img)

axcontrast = fig.add_axes([0.25, 0.1, 0.65, 0.03])
contrast_slider = Slider(
    ax=axcontrast,
    label='Contrast',
    valmin=-1.0,
    valmax=1.0,
    valinit=CONTRAST
)

axbrightness = fig.add_axes([0.25, 0.2, 0.65, 0.03])
brightness_slider = Slider(
    ax=axbrightness,
    label='Brightness',
    valmin=-1.0,
    valmax=1.0,
    valinit=BRIGHTNESS
)

saveax = fig.add_axes([0.8, 0.025, 0.1, 0.04])
button = Button(saveax, 'Save', hovercolor='0.975')

def draw(val):
  global img, img2
  CONTRAST = contrast_slider.val
  BRIGHTNESS = brightness_slider.val
  img2 = np.array(img) / 255
  img2 = 0.5 + (img2 - 0.5) * (1.0 + CONTRAST)
  img2 = np.clip(img2 + BRIGHTNESS, 0.0, 1.0)
  img2 = Image.fromarray(np.uint8(img2 * 255))
  ax[1].imshow(img2)
  fig.canvas.draw_idle()

def save(event):
  global img2
  img2.save("lektion6/bearbeitet.jpeg")

contrast_slider.on_changed(draw)
brightness_slider.on_changed(draw)
button.on_clicked(save)

plt.show()