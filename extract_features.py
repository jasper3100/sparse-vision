import torch 
import os
import h5py

from main import layer_name, original_activations_folder_path
from model import model
from data import input_data
from auxiliary_functions import get_names_of_all_layers, store_feature_maps

'''
Extract features of all layers from a specific layer onwards and store them.
For comparing the feature maps of the original and modified model, only the
layers from the specified layer (layer_name) onwards are relevant.
'''

if __name__ == "__main__":

    model.eval()

    # Placeholder to store activations
    activations = {}

    # Define a hook to capture activations
    def hook(module, input, output, name):
        if name not in activations:
            activations[name] = []
        activations[name].append(output)

    modified_layer_names = get_names_of_all_layers(model)
    # Get layer names from the specified layer onwards
    trunc_layer_names = modified_layer_names[modified_layer_names.index(layer_name):]
    # TO-DO: If we don't want to store the activations of all layers, then we can 
    # reduce the number of layers here. F.e. see "store_feature_maps of all layers using hooks.py"
    # to get only the higher-level modules of the model

    # attach the hooks, from the specified layer onwards
    for name in trunc_layer_names:
        exec(f"{name}.register_forward_hook(lambda module, inp, out, name=name: hook(module, inp, out, name))")

    # Forward pass through the model
    with torch.no_grad():
        output = model(input_data)
        # Once we perform the forward pass, the hook will store the activations 
        # in the dictionary, which is called 'intermediate_activations'

    # Save the activations
    store_feature_maps(
        trunc_layer_names, 
        activations, 
        original_activations_folder_path)