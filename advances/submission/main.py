from diffusers import DiffusionPipeline
from diffusers import StableDiffusionInpaintPipeline
from tqdm import tqdm
import os
import torch
import numpy as np
from PIL import Image, ImageFilter
import cv2
import math
from matplotlib import pyplot as plt
import argparse

def safety_checker(images, clip_input):
    return images, False

def generate_outpainting(zoom_factor, prompt, images_to_generate = 64, start_image=None, folder="."):
  prefix = "outpainting"
  # Target Size after resizing
  original_size = (512, 512)
  target_size = (int(512 / zoom_factor), int(512 / zoom_factor))
  sx = int(((original_size[0]-target_size[0])/2))
  ex = int(((original_size[0]+target_size[0])/2))
  sy = int(((original_size[1]-target_size[1])/2))
  ey = int(((original_size[1]+target_size[1])/2))

  # Generate the initial image
  if start_image is None:
    image = Image.open(os.path.join(folder, "prime_0.png"))
  
  image.save(os.path.join(folder, "{}_0.png".format(prefix)))

  # Setup Inpainting Pipeline
  pipe = StableDiffusionInpaintPipeline.from_pretrained(
      "runwayml/stable-diffusion-inpainting",
      revision="fp16",
      torch_dtype=torch.float16,
  )
  pipe.safety_checker = safety_checker
  pipe.to("cuda")

  N = images_to_generate
  iteration = 1
  while iteration < N:
    # Rescale the image
    resized = image.resize(target_size)

    mask = np.ones(original_size, dtype=np.float32) * 255
    mask[sx:ex, sy:ey] = 0#(pic > .025)*255
    mask = Image.fromarray(mask)
    
    image2 = np.zeros_like(image, dtype=np.float32)
    smooth = image.filter(ImageFilter.GaussianBlur(15))
    smooth = np.array(smooth.getdata()).reshape((original_size[0],original_size[1],3))
    image2[0:original_size[0],0:original_size[1]] = smooth
    
    image2[sx:ey, sy:ey] = resized
    image2 = Image.fromarray(image2.astype(np.uint8))
    while True:
      #image2.save(os.path.join(folder,"./{}_{}_primer.png".format(prefix, iteration)))
      image2 = pipe(prompt=prompt, image=image2, mask_image=mask).images[0]
      
      imgnp = np.array(image2.getdata()).reshape(image.size[0], image.size[1], 3)
      if np.max(imgnp) < 1:
        print("image was NSFW, repeating")
      else:
        break
    
    image2.save(os.path.join(folder,"./{}_{}.png".format(prefix, iteration)))
    image = image2
    iteration = iteration + 1

def smooth_zoom(frm, to, factor, steps, offset, folder):
  for step in range(steps):
    ratio = step / (steps)
    # Create a full sized version merging both images
    full_size = (int(to.size[0] * factor), int(to.size[1] * factor))
    full = to.resize(full_size)
    sx = int(((full.size[0]-frm.size[0])/2))
    ex = int(((full.size[0]+frm.size[0])/2))
    sy = int(((full.size[1]-frm.size[1])/2))
    ey = int(((full.size[1]+frm.size[1])/2))
    full = np.array(full.getdata()).reshape(full.size[0], full.size[1], 3)
    frmnp = np.array(frm.getdata()).reshape(frm.size[0], frm.size[1], 3)
    full[sx:ex,sy:ey] = ratio * full[sx:ex,sy:ey] + (1.0 - ratio) * frmnp
    full = Image.fromarray(full.astype(np.uint8))
    #full.show()
    
    # Scale it
    f = math.pow(1.0 / factor, ratio)
    target_size = ((int)(full_size[0] * f), (int)(full_size[1] * f))
    full = full.resize(target_size)
    full = np.array(full.getdata()).reshape(full.size[0], full.size[1], 3)
    center = full[(int)(full.shape[0]/2-256):(int)(full.shape[0]/2+256),
                  (int)(full.shape[1]/2-256):(int)(full.shape[1]/2+256)]

    center = Image.fromarray(center.astype(np.uint8))

    center.save(os.path.join(folder, "smooth_{}.png".format(step + offset)))
    # if step == int(steps/2):
    #   plt.imshow(center)
    #   plt.pause(0.02)

def smooth_zoom_all(steps, zoom_factor, images, folder):
  for index in tqdm(range(0,images-1)):
    frm = Image.open(os.path.join(folder, "outpainting_{}.png".format(index)))
    to = Image.open(os.path.join(folder, "outpainting_{}.png".format(index+1)))

    smooth_zoom(frm, to, zoom_factor, steps, index*steps, folder)

def create_video(video_name, total_images, zoom_in=True, folder="."):
  images =["smooth_{}.png".format(idx) for idx in range(0,total_images,1)]
  images.reverse()
  frame = cv2.imread(os.path.join(folder, images[0]))
  height, width, layers = frame.shape

  video = cv2.VideoWriter(video_name, 0, 30, (width,height))

  for image in images:
      video.write(cv2.imread(os.path.join(folder, image)))

  cv2.destroyAllWindows()
  video.release()

def generate_prime(prompt, folder):
    pipeline = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
    pipeline.safety_checker = safety_checker
    pipeline.to("cuda")

    while True:
      image = pipeline(prompt).images[0]
      
      imgnp = np.array(image.getdata()).reshape(image.size[0], image.size[1], 3)
      if np.max(imgnp) < 1:
        print("image was NSFW, repeating")
      else:
        break

    image.save(os.path.join(folder, "prime_0.png"))

def infinite_zoom(zoom_factor, prompt, prefix, outpaintings=80, resize_steps = 24, zoom_in = True):
  try:
    os.mkdir(os.path.join('.', prefix))
  except:
    pass
  
  generate_prime(prompt=prompt, folder=prefix)
  generate_outpainting(zoom_factor = zoom_factor, prompt=prompt, images_to_generate=outpaintings, folder=prefix)
  smooth_zoom_all(steps = resize_steps, zoom_factor=zoom_factor, images=outpaintings,folder=prefix)
  create_video(video_name=prefix+".avi", total_images=(outpaintings-1)*resize_steps-1, zoom_in=zoom_in, folder=prefix)

parser = argparse.ArgumentParser(
                    prog='Infinity',
                    description='Create and infinite zoom video based on generated image outpaintings',
                    epilog='To infinity and beyond [Buzz Lightyear, 1995]')

parser.add_argument("-p", "--prompt", default="dwarven village with camp fire")
parser.add_argument("-o", "--paintings", default=40)
parser.add_argument("-s", "--steps", default=16)
parser.add_argument("-z", "--zoom", default=1.6)
args = parser.parse_args()
#"zoo with rich vegetation and various wild life"
infinite_zoom(zoom_factor=args.zoom, prompt=args.prompt, prefix="cache", outpaintings=args.paintings, resize_steps=args.steps)

