import torch
import os
from torchvision.io import read_image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import make_grid
from tqdm import tqdm
import random
from enum import Enum
from device import DEVICE

TRAIN_SET_FOLDER = "d:\\data\\catsvsdogs\\train\\"
TEST_SET_FOLDER = "d:\\data\\catsvsdogs\\test1\\"

resize_transform = transforms.Compose([
    transforms.Resize((160, 160), antialias=True),
])

class DataSetMode(Enum):
    Training = 1,
    Validation = 2,
    Test = 3,
    Final = 4

class CatsDogsDataSet(Dataset):
    def __init__(self, folder, max_samples_per_class = None, transform=None, mode = DataSetMode.Training):
        self.data = []
        self.transform = transform
        cats = []
        dogs = []

        # Create two lists of all relevant files, one for cats, one for dogs
        for _, _, files in os.walk(folder):
            bar = tqdm(files, desc="Scanning images")
            for name in bar:
                path = os.path.join(folder, name)
                
                # Images from the test set are named differently, so we need
                # two different ways to determine the labels
                if not mode == DataSetMode.Test:
                  # In the training set, images are named cat.xxx or dog.xxx
                  # so we split at "." and take the second entry
                  number = int(name.split('.')[1])

                  # We split 80%/20% between training and validation set 
                  # to get an independent estimate for our accuracy during training
                  if mode == DataSetMode.Validation:
                      if number % 5 != 0:
                          continue
                  elif mode == DataSetMode.Training:
                      if number % 5 == 0:
                          continue
                else:
                  # For the test set, we directly append to the data using the image index as the label
                    number = int(name.split('.')[0])
                    self.data.append((path, number-1))

                # We can extract the class by checking whether the name starts with "dog"
                if name.startswith("dog"):
                    dogs.append(path)
                else:
                    cats.append(path)

        # If this is a test set, we are done (we have already appended everything into data)
        if mode == DataSetMode.Test:
          return
        
        
        # Restrict the data we actually load
        if max_samples_per_class is not None:
            # Shuffle both lists, so we always pick a random subset
            random.shuffle(cats)
            random.shuffle(dogs)
        
            # Now take the first n samples
            cats = cats[:max_samples_per_class]
            dogs = dogs[:max_samples_per_class]

        # Append the cats with label 0
        for path in cats:
            self.data.append((path, 0))
        
        # Append the dogs with label 1
        for path in dogs:
            self.data.append((path, 1))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
      # Get the path and label
      path, label = self.data[idx]

      # Read image from disk and move the GPU
      image = read_image(path).to(DEVICE)

      # Resize to target size
      image = resize_transform(image)
      label = torch.Tensor([label]).type(torch.LongTensor).to(DEVICE).reshape(-1)

      # Apply transformation if any
      if self.transform:
          image = self.transform(image)

      # Return the image and the label
      return image, label

if __name__ == "__main__":
    from matplotlib import pyplot as plt
    
    transform = torch.nn.Sequential(
        transforms.Resize((128, 128), antialias=True),
        transforms.ConvertImageDtype(torch.float32)
    )

    dataset = CatsDogsDataSet(TRAIN_SET_FOLDER, max_samples_per_class=20, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    batch, label = dataloader.__iter__().__next__()
    
    grid = make_grid(batch, 8, padding=4).permute(1,2,0)

    plt.imshow(grid.cpu())
    plt.show()

