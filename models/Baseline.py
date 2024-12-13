import torch as th

class Baseline(th.nn.Module):
    def __init__(self):
        super().__init__()

        self.net = th.nn.Sequential(
            th.nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3),
            th.nn.ReLU(),
            th.nn.MaxPool2d(kernel_size=2, stride=2),
            th.nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3),
            th.nn.ReLU(),
            th.nn.MaxPool2d(kernel_size=2, stride=2),
            th.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3),
            th.nn.ReLU(),
            th.nn.Flatten(start_dim=1),
            th.nn.Linear(in_features=32*12*12, out_features=15, bias=True),
        )

    def forward(self, x):
        return self.net(x)