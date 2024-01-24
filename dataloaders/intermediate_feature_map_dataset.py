from torch.utils.data import Dataset
import os

from utils_feature_map import load_feature_map

class IntermediateActivationsDataset(Dataset):
    def __init__(self, layer_name, root_dir, batch_size):
        self.layer_name = layer_name
        self.root_dir = root_dir
        #self.intermediate_activation = None
        self.batch_size = batch_size
        self.image_size = None

    def __len__(self):
        file_path = os.path.join(self.root_dir, 'num_batches.txt')
        with open(file_path, 'r') as file:
            num_batches = int(file.read())
        return num_batches * self.batch_size

    def __getitem__(self, batch_idx):
        file_path = os.path.join(self.root_dir, f'{self.layer_name}_activations_batch_{batch_idx}.h5')
        intermediate_activation = load_feature_map(file_path).float()
        if batch_idx == 1:
            self.image_size = intermediate_activation.shape[1:]
        return intermediate_activation
    
    def get_image_size(self):
        if self.image_size is None:
            # Call __getitem__ to get the image size if it's not set yet
            self.__getitem__(batch_idx=1)
        return self.image_size