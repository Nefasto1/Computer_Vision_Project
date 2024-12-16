import torch as th

class Model(th.nn.Module):
    def __init__(self, kernel_size=3, normalization=False, dropout=False, additional=False):
        super().__init__()

        padding = (kernel_size-1)//2

        self.net = th.nn.Sequential(
            th.nn.Conv2d(in_channels=1, out_channels=8, kernel_size=kernel_size, padding=padding),
            th.nn.Dropout(p=0.5)                                                       if dropout else th.nn.Identity(),
            th.nn.BatchNorm2d(num_features=8)                                         if normalization else th.nn.Identity(),
            th.nn.ReLU(),
            th.nn.MaxPool2d(kernel_size=2, stride=2),
            
            th.nn.Conv2d(in_channels=8, out_channels=16, kernel_size=kernel_size, padding=padding),
            th.nn.Dropout(p=0.5)                                                       if dropout else th.nn.Identity(),
            th.nn.BatchNorm2d(num_features=16)                                         if normalization else th.nn.Identity(),
            th.nn.ReLU(),
            th.nn.MaxPool2d(kernel_size=2, stride=2),
            
            th.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=kernel_size, padding=padding),
            th.nn.Dropout(p=0.5)                                                       if dropout else th.nn.Identity(),
            th.nn.BatchNorm2d(num_features=32)                                         if normalization else th.nn.Identity(),
            th.nn.ReLU(),
            th.nn.MaxPool2d(kernel_size=2, stride=2),
            
            th.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=kernel_size, padding=padding)     if additional else th.nn.Identity(),
            th.nn.Dropout(p=0.5)                                                       if dropout and additional else th.nn.Identity(),
            th.nn.BatchNorm2d(num_features=32)                                         if normalization and additional else th.nn.Identity(),
            th.nn.ReLU()                                                               if additional else th.nn.Identity(),
            
            th.nn.Flatten(start_dim=1),
            th.nn.Linear(in_features=32*8*8, out_features=15, bias=True),
            th.nn.Linear(in_features=15, out_features=15, bias=True)                   if additional else th.nn.Identity(),
        )

    def forward(self, x):
        return self.net(x)