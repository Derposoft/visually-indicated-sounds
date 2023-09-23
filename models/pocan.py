import torch.nn as nn


class POCAN(nn.Module):
    def __init__(self):
        super(POCAN, self).__init__()
        self.fc1 = nn.Linear(in_features=64, out_features=10)

    def forward(self, x):
        x = self.fc1(x)
        return x
