import torch
import h5py

# FIX IMPORTS!!! FROM PARENT DIRECTORY

from sae import SparseAutoencoder
from old_main_with_individual_steps import expansion_factor, sae_weights_file_path, activations_file_path

if __name__ == '__main__':

    # Load the intermediate feature maps, which are the training data for the SAE
    with h5py.File(activations_file_path, 'r') as h5_file:
        data = h5_file['data'][:]

    # convert data to PyTorch tensor
    data = torch.from_numpy(data)

    sae = SparseAutoencoder(input_tensor=data, expansion_factor=expansion_factor)
    sae.load_state_dict(torch.load(sae_weights_file_path))
    sae.eval()

    encoder_output, sae_output = sae(data)

    #print(sae_output)
    #print(sae_output.shape)
    #print(data.shape)

    def measure_sparsity(tensor):
        return torch.sum(tensor == 0).item() / torch.numel(tensor)
    
    def batch_analysis(batch):
        # given a batch of dimension: [batch_size, channels, height, width]
        # enumerate runs over the (# batch_size) feature_maps of dimension: [channels, height, width]
        sparsity = 0
        for i, feature_map in enumerate(batch):
            #print(f"Shape of feature map {i + 1}:", data_sample.shape)
            #print(f"Norm value of feature map {i + 1}: {torch.norm(data_sample).item()}")
            #print(f"Sparsity of feature map {i + 1}: {measure_sparsity(feature_map)}")
            sparsity += measure_sparsity(feature_map)

        print(f"Average sparsity: {sparsity / len(batch)}")

    batch_analysis(data)
    batch_analysis(sae_output)   