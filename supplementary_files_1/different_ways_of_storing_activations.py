from utils import *

# DIFFERENT WAYS OF STORING ACTIVATIONS

# store the activations of layer 'name' for the current batch, this takes 39 seconds for MNIST
store_batch_feature_maps(output, self.dataset_length, name, self.folder_path, self.params)

# the cat operation takes a long time, overall 93 seconds for MNIST
if name not in self.activations:
    self.activations[name] = output 
else:
    self.activations[name] = torch.cat((self.activations[name], output), dim=0)

# store the activations, this takes 28 seconds for MNIST
if name not in self.activations:
    self.activations[name] = []
self.activations[name].append(output)