import os
import h5py
import torch

def store_feature_maps(layer_names, activations, folder_path):
    # create the folder
    os.makedirs(folder_path, exist_ok=True)

    # store the intermediate feature maps
    for name in layer_names:
        activation = activations[name]

        # if activation is empty, we give error message
        if activation.nelement() == 0:
            raise ValueError(f"Activation of layer {name} is empty")
        else:
            activations_file_path = os.path.join(folder_path, f'{name}_activations.h5')
            # Store activations to an HDF5 file
            with h5py.File(activations_file_path, 'w') as h5_file:
                h5_file.create_dataset('data', data=activation.numpy())

def load_feature_map(file_path):
    with h5py.File(file_path, 'r') as h5_file:
        data = torch.from_numpy(h5_file['data'][:])
    # if data is emtpy, we give error message
    if data.nelement() == 0:
        raise ValueError(f"Trying to load feature map but it is empty")
    else:
        return data