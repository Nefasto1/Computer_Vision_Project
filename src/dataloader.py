import torch as th
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
from torchvision.io import read_image

import os

import numpy as np

class CustomDataset(Dataset):
    def __init__(self, train=True, transform=None, X=None, y=None):
        if X is None or y is None:
            self.X, self.y = self._extract_images(train)
        else:
            self.X, self.y = X, y
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

def customDataloader(transform=None, batch_size=64, shuffle=True):
    dataset      = CustomDataset(train=True,  transform=None)

    n              = len(dataset)
    train_idx      = np.random.choice(range(n), size=int(n*0.85), replace=False)
    validation_idx = set(range(n)) - set(train_idx) 

    X_train, y_train = dataset[train_idx]
    X_val, y_val     = dataset[list(validation_idx)]
    
    train_dataset      = CustomDataset(X=X_train, y=y_train, transform=transform)
    validation_dataset = CustomDataset(X=X_val,   y=y_val,   transform=transform)
    test_dataset       = CustomDataset(train=False, transform=None)

    train_dataloader      = DataLoader(train_dataset,      batch_size=batch_size, shuffle=shuffle)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=shuffle)
    test_dataloader       = DataLoader(test_dataset,       batch_size=batch_size, shuffle=shuffle)

    return train_dataloader, validation_dataloader, test_dataloader
