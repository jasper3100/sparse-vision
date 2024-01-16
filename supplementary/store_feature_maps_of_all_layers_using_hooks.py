# CODE TO STORE THE ACTIVATIONS OF ALL LAYERS OF A MODEL USING HOOKS

from torchvision.models import resnet50, ResNet50_Weights
import torch
import torch.nn as nn
import re

from auxiliary_functions import get_names_of_all_layers

# Load the pre-trained model
weights = ResNet50_Weights.IMAGENET1K_V2
model = resnet50(weights=weights)
model.eval()

# Sample input data (replace this with your image processing code)
input_data = torch.randn(1, 3, 224, 224)  # Example input data with shape (batch_size, channels, height, width)

'''
We want to do model.layer1[0].conv1.register_forward_hook(lambda module, inp, out: hook(module, inp, out, 'layer1_0_conv1'))
But the names of the modules come as layer1.0.conv1,... We need to convert this to model.layer1[0].conv1
'''

# Placeholder to store intermediate activations
intermediate_activations = {}

# Define a hook to capture intermediate activations
def hook(module, input, output, name):
    if name not in intermediate_activations:
        intermediate_activations[name] = []
    intermediate_activations[name].append(output)

modified_module_names = get_names_of_all_layers(model)
#print(modified_names)
#print(model) 
#print(model.children()) only gets layer1, layer2,... but not the sublayers

'''
If we only care about higher level modules of a model we can use the following
code snippet to get names and the modules themselves as well.
F.e. modules of ResNet50 are: conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4, avgpool, fc

def summarize_model(model):
    names = []
    modules = []
    for name, module in model.named_children():
        names.append(name)
        modules.append(module)
    return names, modules

names, modules = summarize_model(model)
# Add "model." to the beginning of each name
names = [f"model.{name}" for name in names]
print(names)
'''

# attach the hooks
for name in modified_module_names:
    exec(f"{name}.register_forward_hook(lambda module, inp, out, name=name: hook(module, inp, out, name))")

# Forward pass through the model
with torch.no_grad():
    model(input_data)
    # Once we perform the forward pass, the hook will store the activations 
    # in the dictionary, which is called 'intermediate_activations'

########################################################################################
# HOW TO ACCESS THE INTERMEDIATE FEATURE MAPS

# We can specify the desired layer of which we want to access the feature maps by name or by index
# In ResNet50, there are 126 layers in total --> indices: 0, ..., 125
layer_name = 'model.layer1' #'model.layer1[0].downsample[1]' 
#layer_index = [SPECIFY INDEX HERE]

# Get index, given name
layer_index = modified_module_names.index(layer_name)
# Get name, given index
layer_name = modified_module_names[layer_index]

activation = intermediate_activations[layer_name][0] # we do [0], because the tensor that we want is inside of a list
#print(activation.shape)