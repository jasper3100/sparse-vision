import torch
import h5py
import os

from sae import SparseAutoencoder
from main import sae_weights_folder_path, expansion_factor, layer_name, adjusted_activations_folder_path
from data import input_data
from model import model
from auxiliary_functions import get_names_of_all_layers, store_feature_maps

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
        sae_weights_file_path = os.path.join(sae_weights_folder_path, f'{layer_name}_trained_sae_weights.pth')
        sae.load_state_dict(torch.load(sae_weights_file_path))
        sae.eval()
        _, modified_output = sae(layer_output)
        return modified_output

    activations = {}

    def hook(module, input, output, name):
        # modify output of the specified layer
        if name == layer_name:
            output[0] = modify_layer_output(output[0])
            print(f"Modified output of layer {name}")
        # store the activations
        if name not in activations:
            activations[name] = []
        activations[name].append(output)

    modified_layer_names = get_names_of_all_layers(model)
    # Get layer names from the specified layer onwards
    trunc_layer_names = modified_layer_names[modified_layer_names.index(layer_name):]

    # attach the hooks 
    for name in trunc_layer_names:
        exec(f"{name}.register_forward_hook(lambda module, inp, out, name=name: hook(module, inp, out, name))")
        
    # Forward pass through the model
    with torch.no_grad():
        output = model(input_data)
        # During the forward pass, the hook will adjust the output 
        # of the chosen layer and continue with the forward pass
    
    # Save the activations
    store_feature_maps(
        trunc_layer_names, 
        activations, 
        adjusted_activations_folder_path)