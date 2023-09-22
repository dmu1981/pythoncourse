from torch.utils.data import Dataset

class DatasetCache(Dataset):
    def __init__(self, dataset, transform = None):
        self.dataset = dataset
        self.transform = transform
        self.cache = [None for _ in range(len(self.dataset))]

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        item = self.cache[index]
        if item is None:
            item = self.dataset[index]
            self.cache[index] = item

        if self.transform is not None:
            image, label = item
            image = self.transform(image)
            item = (image, label)

        return item