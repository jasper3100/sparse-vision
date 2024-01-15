import torch 
import json
import os
import numpy as np
import h5py  
import time

# TO-DO: IMPORT THESE THINGS PROPERLY

from code.main import layer_name, activations_folder_path, activations_file_path
#..main import layer_name, activations_folder_path, activations_file_path
from ..model import model, weights
from ..data import input_data
from ..auxiliary_functions import print_result

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
    
    # JSON
    start_time = time.time()
    # Convert PyTorch tensor to NumPy array before saving to JSON
    numpy_activation = activation.numpy()
    # Store intermediate_activations to a JSON file in the specified folder
    with open(activations_file_path, 'w') as json_file:
        # Convert NumPy array to list to make it JSON serializable
        json.dump(numpy_activation.tolist(), json_file)
    end_time = time.time()
    print(f"Time taken to store tensor in JSON: {end_time - start_time} seconds")

    # HDF5
    start_time = time.time()
    # Store intermediate_activations to an HDF5 file (or any other format)
    with h5py.File(activations_file_path, 'w') as h5_file:
        h5_file.create_dataset('data', data=activation.numpy())
    end_time = time.time()
    print(f"Time taken to store tensor HDF5: {end_time - start_time} seconds")

    # NPZ
    start_time = time.time()
    np.savez_compressed(activations_file_path, data=activation.numpy())
    end_time = time.time()
    print(f"Time taken to store tensor NPZ: {end_time - start_time} seconds")