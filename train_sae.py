import torch
import os
import numpy as np
import h5py

from sae import SparseAutoencoder
from sparse_loss import SparseLoss
from main import activations_file_path, sae_weights_folder_path, sae_weights_file_path, expansion_factor

if __name__ == "__main__":

    # Load the intermediate feature maps, which are the training data for the SAE
    with h5py.File(activations_file_path, 'r') as h5_file:
        data = h5_file['data'][:]

    # convert data to PyTorch tensor
    data = torch.from_numpy(data)

    # Instantiate the sparse autoencoder (SAE) model
    sae = SparseAutoencoder(data, expansion_factor)
    optimizer = torch.optim.Adam(sae.parameters(), lr=0.001)
    criterion = SparseLoss(lambda_sparse=0.1)

    # Making sure that model and data have same numeric format
    sae = sae.float()
    data = data.float()

    # Training loop
    num_epochs = 3 # custom choice TO DO MAKE THIS A PARAMETER!!!

    for epoch in range(num_epochs):
        sae.train()
        optimizer.zero_grad()
        encoded, decoded = sae(data)
        loss = criterion(encoded, decoded, data)
        loss.backward()
        optimizer.step()
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

    # Ensure the folder exists; create it if it doesn't
    os.makedirs(sae_weights_folder_path, exist_ok=True)

    torch.save(sae.state_dict(), sae_weights_file_path)