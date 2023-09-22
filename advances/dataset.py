import torch
import os
from torchvision.io import read_image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import make_grid
from tqdm import tqdm
import random
from device import DEVICE
TRAIN_SET_FOLDER = "f:\\data\\catsvsdogs\\train\\"

resize_transform = transforms.Compose([
    transforms.Resize((160, 160), antialias=True),
])

class CatsDogsDataSet(Dataset):
    def __init__(self, folder, max_samples_per_class = None, transform=None, is_validation=False):
        self.data = []
        cats = []
        dogs = []

        # Create two lists of all relevant files, one for cats, one for dogs
        for _, _, files in os.walk(folder):
            bar = tqdm(files, desc="Scanning images")
            for name in bar:
                path = os.path.join(folder, name)

                number = int(name.split('.')[1])
                if is_validation:
                    if number % 5 != 0:
                        continue
                else:
                    if number % 5 == 0:
                        continue


                if name.startswith("dog"):
                    dogs.append(path)
                else:
                    cats.append(path)

        # Shuffle both lists
        random.shuffle(cats)
        random.shuffle(dogs)

        # Restrict the data we actually load
        if max_samples_per_class is not None:
            cats = cats[:max_samples_per_class]
            dogs = dogs[:max_samples_per_class]

        for path in cats:
            self.data.append((path, 0))
        
        for path in dogs:
            self.data.append((path, 1))

        # Note: We don´t need to shuffle here as the
        # data loader will do the shuffling for us
                
        self.transform = transform


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
      path, label = self.data[idx]
      image = read_image(path).to(DEVICE)
      image = resize_transform(image)
      label = torch.Tensor([label]).type(torch.LongTensor).to(DEVICE).reshape(-1)

      if self.transform:
          image = self.transform(image)

      return image, label

if __name__ == "__main__":
    from matplotlib import pyplot as plt
    
    transform = torch.nn.Sequential(
        transforms.Resize((64, 64), antialias=True),
        transforms.ConvertImageDtype(torch.float32)
    )

    dataset = CatsDogsDataSet(TRAIN_SET_FOLDER, max_samples_per_class=20, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    batch, label = dataloader.__iter__().__next__()
    #batch = torch.Tensor(batch)
    
    grid = make_grid(batch, 8, padding=4).permute(1,2,0)

    plt.imshow(grid.cpu())
    plt.show()

