import torch
import torch.nn as nn

'''
Small check to see if the hook manipulates a layers' output in the expected way.

Here, we define a small model with 2 linear layers. The second layer has input size 2 and output size 1. We specify the weights
as 2 and 5 and the bias as 3. Thus, if we set the output of the first layer to [1,1] using the hook, then the expected output
is 10. This is indeed the case. 

In addition, we check for a larger (but still small) model that using two hooks and removing a hook also work (disregarding the exact values).
'''

# keep randomness of input and model weight initialization constant across different runs
torch.manual_seed(0)

class SmallModel(nn.Module):
    def __init__(self):
        super(SmallModel, self).__init__()
        self.fc1 = nn.Linear(3, 2)
        self.fc2 = nn.Linear(2, 1)

        # specify custom weights and bias of second layer
        with torch.no_grad():
            self.fc2.weight.copy_(torch.tensor([[2, 5]], dtype=torch.float32))
            self.fc2.bias.copy_(torch.tensor([3], dtype=torch.float32))

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

# Instantiate the small model
model = SmallModel()
model.eval()

# Define a modification function for the layer output
def modify_layer_output(output_tensor):
    return torch.ones_like(output_tensor) # * 2, another variation to verify that the hook works as intended

def modify_output(name, modification):
    def hook(module, input, output):
        output[0] = modification(output[0])  # Modify the output tensor of the layer
    return hook

# Register the hook to the desired layer along with the modification function
model.fc1.register_forward_hook(modify_output('fc1', modify_layer_output))
# or more generally: exec(f"{layer_name}.register_forward_hook(modify_output('{layer_name}', modify_layer_output))")

input_data = torch.randn(1, 3)

# print result of forward pass, it is indeed 10 as expected
print('Result of linear model: {}, as expected'.format(model(input_data)))

##############################################################################################################
# In addition, we check that for another small model, the hook changes the output of the model (the exact
# values are not verified)

# Construct a small model with 2 convolutional layers performing classification, input shape is [1,3,224,224]
class SmallModel2(nn.Module):
    def __init__(self):
        super(SmallModel2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32 * 224 * 224, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x
    
# Instantiate the small model
model2 = SmallModel2()
model2.eval()

input_data2 = torch.randn(1, 3, 224, 224)

def modify_layer_output2(output_tensor):
    return output_tensor + 1

print('Result of convolutional model before modification: {}'.format(model2(input_data2)))

hook = model2.conv1.register_forward_hook(modify_output('conv1', modify_layer_output2))
print('Result of convolutional model after registering hook on conv1 layer: {} --> different than before, as desired'.format(model2(input_data2)))

hook.remove()
print('Result of convolutional model after removing hook on conv1 layer: {} --> back to original output, as desired'.format(model2(input_data2)))

model2.conv2.register_forward_hook(modify_output('conv2', modify_layer_output2))
print('Result of convolutional model after registering hook on conv2 layer: {} --> different than before, as desired'.format(model2(input_data2)))

model2.conv1.register_forward_hook(modify_output('conv1', modify_layer_output2))
print('Result of convolutional model after registering hook on conv1 layer on top: {} --> different than before, as desired'.format(model2(input_data2)))
