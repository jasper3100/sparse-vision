import torch.nn as nn
import torch.nn.functional as F
import torch

'''
IT IS IMPORTANT THAT EACH LAYER IS ONLY USED ONCE IN THE MODEL. 
For instance, if we do: self.act = nn.ReLU() and then in the forward
method: x = self.act(self.fc1(x)) and x = self.act(self.fc2(x)), then
the activations of the act layer will contain activations from 2 different
places, which will lead to confusions!
'''

class CustomMLP1(nn.Module):
    def __init__(self, img_size):
        # call the constructor of the parent class
        super(CustomMLP1, self).__init__()
        self.img_size = img_size
        self.prod_size = torch.prod(torch.tensor(self.img_size)).item()

        # define the layers
        self.fc1 = nn.Linear(self.prod_size, 256)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 256)
        self.act2 = nn.ReLU()
        self.fc3 = nn.Linear(256, 10)
        self.sm = nn.Softmax(dim=1)

    def forward(self, x):
        # flatten the input
        x = x.view(-1, self.prod_size)
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        x = self.sm(self.fc3(x))
        return x