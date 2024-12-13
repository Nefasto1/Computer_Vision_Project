import torch as th

class Model(th.nn.Module):
    def __init__(self, kernel_size=3, normalization=False, dropout=False):
        super().__init__()

        out_size = (64 - kernel_size + 1)//2
        out_size = (out_size - kernel_size + 1)//2
        out_size = (out_size - kernel_size + 1)

        self.net = th.nn.Sequential(
            th.nn.Conv2d(in_channels=1, out_channels=8, kernel_size=kernel_size),
            th.nn.Dropout(p=0.5) if dropout else th.nn.Identity(),
            th.nn.BatchNorm2d(num_features=8) if normalization else th.nn.Identity(),
            th.nn.ReLU(),
            th.nn.MaxPool2d(kernel_size=2, stride=2),
            th.nn.Conv2d(in_channels=8, out_channels=16, kernel_size=kernel_size),
            th.nn.Dropout(p=0.5) if dropout else th.nn.Identity(),
            th.nn.BatchNorm2d(num_features=16) if normalization else th.nn.Identity(),
            th.nn.ReLU(),
            th.nn.MaxPool2d(kernel_size=2, stride=2),
            th.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=kernel_size),
            th.nn.Dropout(p=0.5) if dropout else th.nn.Identity(),
            th.nn.BatchNorm2d(num_features=32) if normalization else th.nn.Identity(),
            th.nn.ReLU(),
            th.nn.Flatten(start_dim=1),
            th.nn.Linear(in_features=32*out_size*out_size, out_features=15, bias=True),
        )

    def forward(self, x):
        return self.net(x)