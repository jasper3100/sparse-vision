import torch 
import os
import h5py

from main import layer_name, original_activations_folder_path
from model import model
from data import input_data
from auxiliary_functions import store_feature_maps, names_of_main_modules_and_specified_layer

'''
Extract features of all layers from a specific layer onwards and store them.
For comparing the feature maps of the original and modified model, only the
layers from the specified layer (layer_name) onwards are relevant, we choose
some of them because otherwise they required storage space would be too high.
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

    # get the names of the main modules of the model and include layer_name
    module_names = names_of_main_modules_and_specified_layer(model, layer_name)
    # This list of module names can be adjusted as desired, i.e., removing/adding layers

    '''
    If we want to use all layers after the specified layer, we can do 
    from auxiliary_functions import get_names_of_all_layers
    modified_layer_names = get_names_of_all_layers(model)
    trunc_layer_names = modified_layer_names[modified_layer_names.index(layer_name):]    
    '''

    # attach the hooks, from the specified layer onwards
    for name in module_names:
        exec(f"{name}.register_forward_hook(lambda module, inp, out, name=name: hook(module, inp, out, name))")

    # Forward pass through the model
    with torch.no_grad():
        output = model(input_data)
        # Once we perform the forward pass, the hook will store the activations 
        # in the dictionary, which is called 'intermediate_activations'

    # Save the activations
    store_feature_maps(
        module_names, 
        activations, 
        original_activations_folder_path)