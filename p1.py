import torch
import torch.nn as nn
import os

from lucent.optvis import render, param, transform, objectives
from lucent.modelzoo.util import get_model_layers

from torchvision.models import resnet18, ResNet18_Weights

model_ft = resnet18()
# Adjust final layers 
model_ft.avgpool = nn.AdaptiveAvgPool2d(1)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 200) 
# Adjust layers at beginning
model_ft.conv1 = nn.Conv2d(3, 64, kernel_size=(3,3), stride=(1,1), padding=(1,1))
model_ft.maxpool = nn.Sequential() # nn.Identity() # remove maxpool layer

# creat a small MLP class
class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, output_size)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
# insert a new layer
model_ft.fc = nn.Sequential(nn.Linear(num_ftrs, 200), nn.ReLU(), MLP(200, 200))

path = "/lustre/home/jtoussaint/master_thesis/model_weights/resnet18_2/tiny_imagenet/None_resnet18_2_10_0.001_100_sgd_w_scheduler_0.1_model_weights.pth"
# if this path exists, load the weights
if os.path.exists(path):
    model_ft.load_state_dict(torch.load(path))
# else, if we are on local machine, just use random weights


print(get_model_layers(model_ft))



#render.render_vis(model_ft, "layer1_0_conv1:45")#, show_inline=True)