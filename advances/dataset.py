import torch
import os
from torchvision.io import read_image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import make_grid
from tqdm import tqdm
import random
from device import DEVICE

TRAIN_SET_FOLDER = "D:\\data\\catsvsdogs\\train\\"

class CatsDogsDataSet(Dataset):
    def __init__(self, folder, max_samples_per_class = None, transform=None):
        self.data = []
        cats = []
        dogs = []

        # Create two lists of all relevant files, one for cats, one for dogs
        for _, _, files in os.walk(folder):
            bar = tqdm(files, desc="Scanning images")
            for name in bar:
                path = os.path.join(folder, name)

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

        for path in tqdm(cats, desc="Loading cats"):
            self.data.append((read_image(path), 0))
        
        for path in tqdm(dogs, desc="Loading dogs"):
            self.data.append((read_image(path), 0))

        # Note: We don´t need to shuffle here as the
        # data loader will do the shuffling for us
                
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx]

        if self.transform:
            image = self.transform(image)

        return image, label    

if __name__ == "__main__":
    from matplotlib import pyplot as plt
    
    transform = torch.nn.Sequential(
        transforms.Resize((256, 256)),
    )

    dataset = CatsDogsDataSet(TRAIN_SET_FOLDER, max_samples_per_class=200, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    batch, label = dataloader.__iter__().__next__()
    batch = torch.Tensor(batch)
    
    grid = make_grid(batch, 8, padding=4).permute(1,2,0)

    plt.imshow(grid)
    plt.show()

