import torch
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import CatsDogsDataSet, TRAIN_SET_FOLDER
from device import DEVICE
import random



training_transform = transforms.Compose([
        transforms.RandomRotation(15),
        transforms.CenterCrop((144, 144)),
        transforms.RandomCrop((128, 128)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ColorJitter(0.3, 0.3, 0.3),
        transforms.ConvertImageDtype(torch.float32)
        ]
    )

validation_transform = transforms.Compose([
        transforms.CenterCrop((128, 128)),
        transforms.ConvertImageDtype(torch.float32)
        ]
    )

if __name__ == "__main__":
    from matplotlib import pyplot as plt
    
    dataset = CatsDogsDataSet(TRAIN_SET_FOLDER)
    
    image, label = dataset[int(random.uniform(0, len(dataset)))]
    
    batch = None
    for index in range(32):
        transformed = training_transform(image)
        transformed = transformed.reshape(1, 3, 128, 128)
        if batch is None:
            batch = transformed
        else:
            batch = torch.cat((batch, transformed))

    grid = make_grid(batch, 8, padding=4).permute(1,2,0)

    plt.imshow(grid.cpu())
    plt.show()