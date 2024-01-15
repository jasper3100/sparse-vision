import torch 
import os
import h5py

from main import layer_name, activations_folder_path, activations_file_path
from model import model, weights
from data import input_data
from auxiliary_functions import print_result

'''
Extract intermediate features of a specific layer and store them.
'''

if __name__ == "__main__":

    model.eval()

    # Placeholder to store intermediate activations
    intermediate_activations = {}

    # Define a hook to capture intermediate activations
    def hook(module, input, output, name):
        if name not in intermediate_activations:
            intermediate_activations[name] = []
        intermediate_activations[name].append(output)

    # attach the hook
    exec(f"{layer_name}.register_forward_hook(lambda module, inp, out, name=layer_name: hook(module, inp, out, name))")

    # Forward pass through the model
    with torch.no_grad():
        model(input_data)
        # Once we perform the forward pass, the hook will store the activations 
        # in the dictionary, which is called 'intermediate_activations'
        print('Classification results before modification:')
        print_result(model, input_data, weights)

    # access the intermediate feature maps
    activation = intermediate_activations[layer_name][0] # we do [0], because the tensor that we want is inside of a list

    # Ensure the folder exists; create it if it doesn't
    os.makedirs(activations_folder_path, exist_ok=True)
    
    # Store intermediate_activations to an HDF5 file
    with h5py.File(activations_file_path, 'w') as h5_file:
        h5_file.create_dataset('data', data=activation.numpy())