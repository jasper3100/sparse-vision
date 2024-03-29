import torch.nn as nn
import torch
import numpy as np

# I TRIED USING THIS MODEL FOR TINY IMAGENET, BUT IT WAS TOO WEAK (max. 25% top 1 eval accuracy)

'''
IT IS IMPORTANT THAT EACH LAYER IS ONLY USED ONCE IN THE MODEL. 
For instance, if we do: self.act = nn.ReLU() and then in the forward
method: x = self.act(self.fc1(x)) and x = self.act(self.fc2(x)), then
the activations of the act layer will contain activations from 2 different
places, which will lead to confusions!
'''

class CustomCNN1(nn.Module):
    def __init__(self, img_size, num_classes):
        super(CustomCNN1, self).__init__()
        # given input of size (B, C, H, W)
        # the output size of a conv layer is (B, out_channels, H_out, W_out), where
        # H_out = floor[(H_in - kernel_size + 2*padding)/stride] + 1
        # W_out = floor[(W_in - kernel_size + 2*padding)/stride] + 1

        # However, img_size is of shape (C, H, W)!

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        # H_out = floor[(H_in - 3 + 2*1)/1] + 1 = H_in, likewise W_out = W_in
        # --> output of conv1 has size (B, 32, H, W)
        self.pool1 = nn.MaxPool2d(2, 2)
        # output of pool1 has size (B, 32, H/2, W/2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # output of conv2 has size (B, 64, H/2, W/2)
        self.pool2 = nn.MaxPool2d(2, 2)
        # output of pool2 has size (B, 64, H/4, W/4)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # output of conv3 has size (B, 128, H/4, W/4)
        self.pool3 = nn.MaxPool2d(2, 2)
        # output of pool3 has size (B, 128, H/8, W/8)

        # if H/2 and W/2 are not integers, we need to floor the result
        self.H_prod_W = int(np.floor(img_size[-1]/8 * img_size[-2]/8))
        print(self.H_prod_W)
        self.fc1 = nn.Linear(128 * self.H_prod_W, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool1(nn.functional.relu(self.conv1(x)))
        x = self.pool2(nn.functional.relu(self.conv2(x)))
        x = self.pool3(nn.functional.relu(self.conv3(x)))
        # --> x has shape (B, 128, H/8, W/8)
        x = x.view(-1, 128 * self.H_prod_W)
        # --> x has shape (B, 128 * H/8 * W/8)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x