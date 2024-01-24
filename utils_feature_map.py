import os
import h5py
import torch

def store_feature_maps(layer_names, activations, folder_path, batch_idx):
    # create the folder
    os.makedirs(folder_path, exist_ok=True)

    # store the intermediate feature maps
    for name in layer_names:
        activation = activations[name][0] # we do [0], because the tensor that we want is inside of a list

        # if activation is empty, we give error message
        if activation.nelement() == 0:
            raise ValueError(f"Activation of layer {name}, batch {batch_idx} is empty")
        else:
            activations_file_path = os.path.join(folder_path, f'{name}_activations_batch_{batch_idx}.h5')
            # Store activations to an HDF5 file
            with h5py.File(activations_file_path, 'w') as h5_file:
                h5_file.create_dataset('data', data=activation.numpy())

def store_feature_maps(layer_names, activations, folder_path):
    # create the folder
    os.makedirs(folder_path, exist_ok=True)

    # Iterate over layer names
    for name in layer_names:
        # Initialize or load the HDF5 file
        activations_file_path = os.path.join(folder_path, f'{name}_activations.h5')
        with h5py.File(activations_file_path, 'a') as h5_file:
            # Iterate over batches and append activations
            for batch_idx, activation in enumerate(activations[name]):
                # if activation is empty, give an error message
                if activation.nelement() == 0:
                    raise ValueError(f"Activation of layer {name}, batch {batch_idx} is empty")
                else:
                    # Create a dataset or append to an existing dataset
                    dataset_name = 'data' # we store all a
                    if dataset_name not in h5_file:
                        h5_file.create_dataset(dataset_name, data=activation.numpy())
                    else:
                        h5_file[dataset_name].resize((h5_file[dataset_name].shape[0] + activation.shape[0]), axis=0)
                        h5_file[dataset_name][-activation.shape[0]:] = activation.numpy()

# Usage example
# store_feature_maps(layer_names, activations, folder_path)



def load_feature_map(file_path):
    with h5py.File(file_path, 'r') as h5_file:
        data = torch.from_numpy(h5_file['data'][:])
    # if data is emtpy, we give error message
    if data.nelement() == 0:
        raise ValueError(f"Trying to load feature map but it is empty")
    else:
        return data