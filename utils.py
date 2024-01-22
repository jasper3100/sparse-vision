import os
import h5py
import torch
import shutil

from model_loader import ModelLoader
from input_data_loader import InputDataLoader

def load_model(model_name):
    model_loader = ModelLoader(model_name)
    model_loader.load_model()
    return model_loader.model, model_loader.weights

def load_data(dataset_name):
    data_loader = InputDataLoader(dataset_name)
    data_loader.load_data()
    return data_loader.data

def store_feature_maps(layer_names, activations, folder_path):
    # store the intermediate feature maps
    for name in layer_names:
        activation = activations[name][0] # we do [0], because the tensor that we want is inside of a list
        
        # if the folder exists, we remove it, to avoid using outdated feature maps somewhere
        #if os.path.exists(folder_path):
        #    shutil.rmtree(folder_path)
        # Ensure the folder exists; create it if it doesn't
        os.makedirs(folder_path, exist_ok=True)

        activations_file_path = os.path.join(folder_path, f'{name}_activations.h5')
        
        # Store activations to an HDF5 file
        with h5py.File(activations_file_path, 'w') as h5_file:
            h5_file.create_dataset('data', data=activation.numpy())

    print("Successfully stored features in", folder_path)

def load_feature_map(file_path):
    with h5py.File(file_path, 'r') as h5_file:
        data = torch.from_numpy(h5_file['data'][:])
    return data