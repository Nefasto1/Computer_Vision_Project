import torch as th
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
from torchvision.io import read_image

import os

class CustomDataset(Dataset):
    def __init__(self, train=True, transform=None):
        self.X, self.y = self._extract_images(train)
        self.transform = transform

    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        X = self.X[idx]
        if self.transform:
            X = self.transform(X)
        
        return X, self.y[idx]
    
    def _extract_images(self, train):
        classes = os.listdir("train/")
        
        initial_path = "train" if train else "test"
        
        X = th.empty(0, 1, 64, 64)
        Y = []
        
        anisotropic = transforms.Resize((64, 64))
        
        for y in range(len(classes)):
            for image in sorted(os.listdir(f"{initial_path}/{classes[y]}")):
                x = read_image(f"{initial_path}/{classes[y]}/{image}")
                x = anisotropic(x).unsqueeze(0)
                
                X = th.cat((X, x))
                Y.append(y)
        
        Y = th.tensor(Y)
        Y = th.nn.functional.one_hot(Y)
    
        return X, Y

def customDataloader(train=True, transform=None, batch_size=64, shuffle=True):
    dataset = CustomDataset(train, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return dataloader