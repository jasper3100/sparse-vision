from torch.utils.data import Dataset

from utils import load_feature_map, get_file_path

class IntermediateActivationsDataset(Dataset):
    def __init__(self, layer_names, original_activations_folder_path, train_dataset_length, params):
        self.train_dataset_length = train_dataset_length
        self.image_size = None

        if len(layer_names) == 1:
            file_path = get_file_path(folder_path=original_activations_folder_path,  
                                        layer_names=layer_names, 
                                        params=params, 
                                        file_name='activations.h5')
            self.combined_activations = load_feature_map(file_path).float()
            self.image_size = self.combined_activations.shape[1:]
        else:
            # TO-DO: decide how to proceed if we want to apply SAE on several layers
            raise ValueError("Building dataset of activations: Case of several layers not yet supported.")

    def __len__(self):
        return self.train_dataset_length

    def __getitem__(self, sample_idx):
        #print(f"Intermediate activation shape: {self.combined_activations[sample_idx].shape}")
        return self.combined_activations[sample_idx]
    
    def get_image_size(self):
        #print(f"Image size: {self.image_size}")
        return tuple(self.image_size)