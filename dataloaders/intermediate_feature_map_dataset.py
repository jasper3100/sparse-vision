from torch.utils.data import Dataset
import os

from utils import load_feature_map

class IntermediateActivationsDataset(Dataset):
    def __init__(self, layer_name, original_activations_folder_path, train_dataset_length):
        self.layer_name = layer_name
        self.original_activations_folder_path = original_activations_folder_path
        self.train_dataset_length = train_dataset_length
        self.image_size = None

    def __len__(self):
        return self.train_dataset_length

    def __getitem__(self, sample_idx):
        file_path = os.path.join(self.original_activations_folder_path, f'{self.layer_name}_activations.h5')
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