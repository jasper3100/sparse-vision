import torch
import h5py
import os

from sae import SparseAutoencoder
from main import sae_weights_file_path, expansion_factor, layer_name, adjusted_model_output_file_path, adjusted_model_output_folder_path
from auxiliary_functions import print_result
from data import input_data
from model import model, weights

'''
Evaluate model on adjusted features. In particular, we use hooks to pass the output of 
the intermediate layer to the autoencoder and insert the output of the autoencoder into the model.
'''

if __name__ == "__main__":

    # Define a modification function for the layer output
    def modify_layer_output(layer_output): # layer output = intermediate feature map outputted by the respective layer
        # Instantiate the sparse autoencoder (SAE) model and load trained weights
        # shape of layer_output: [channels, height, width] --> no batch dimension
        sae = SparseAutoencoder(input_tensor=layer_output, expansion_factor=expansion_factor)
        sae.load_state_dict(torch.load(sae_weights_file_path))
        sae.eval()
        _, modified_output = sae(layer_output)
        return modified_output

    def modify_output(name, modification):
        def hook(module, input, output):
            output[0] = modification(output[0])  # Modify the output tensor of the layer
        return hook

    # attach the hook
    exec(f"{layer_name}.register_forward_hook(modify_output('{layer_name}', modify_layer_output))")

    # Forward pass through the model
    with torch.no_grad():
        model(input_data)
        # During the forward pass, the hook will adjust the output 
        # of the chosen layer and continue with the forward pass
    
    # Store model output to an HDF5 file
    os.makedirs(adjusted_model_output_folder_path, exist_ok=True)
    with h5py.File(adjusted_model_output_file_path, 'w') as h5_file:
        h5_file.create_dataset('data', data=output.numpy())