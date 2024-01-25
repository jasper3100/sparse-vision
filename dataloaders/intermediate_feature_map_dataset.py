from torch.utils.data import Dataset
import os

from utils import load_feature_map

class IntermediateActivationsDataset(Dataset):
    def __init__(self, layer_name, root_dir, batch_size):
        self.layer_name = layer_name
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.image_size = None

    def __len__(self):
        file_path = os.path.join(self.root_dir, 'num_batches.txt')
        with open(file_path, 'r') as file:
            num_batches = int(file.read())
        return num_batches * self.batch_size

    def __getitem__(self, sample_idx):
        file_path = os.path.join(self.root_dir, f'{self.layer_name}_activations.h5')
        combined_activations = load_feature_map(file_path).float()
        if sample_idx == 1:
            self.image_size = combined_activations.shape[1:]
        #print(f"Intermediate activation shape: {combined_activations[sample_idx].shape}")
        return combined_activations[sample_idx]
    
    def get_image_size(self):
        if self.image_size is None:
            self.__getitem__(sample_idx=1)
        #print(f"Image size: {self.image_size}")
        return tuple(self.image_size)