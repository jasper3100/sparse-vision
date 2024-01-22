import torch.nn as nn
import torch.nn.functional as F

class CustomMLP1(nn.Module):
    def __init__(self):
        # call the constructor of the parent class
        super(CustomMLP1, self).__init__()
        self.fc1 = nn.Linear(3*32*32, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 10)
        self.act = nn.ReLU()
        self.sm = nn.Softmax(dim=1)

    def forward(self, x):
        # flatten the input
        x = x.view(-1, 3*32*32)
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.sm(self.fc3(x))
        return x