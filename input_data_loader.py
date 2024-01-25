import torch
import torchvision
import torchvision.transforms as transforms
import os
from torch.utils.data import DataLoader

from dataloaders.tiny_imagenet import TinyImageNetDataset, TinyImageNetPaths
from dataloaders.intermediate_feature_map_dataset import IntermediateActivationsDataset
from model_loader import ModelLoader
from utils_feature_map import load_feature_map

class InputDataLoader:
    def __init__(self, dataset_name, batch_size):
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.img_size = None
        self.train_dataloader = None
        self.val_dataloader = None

    def load_tiny_imagenet(self, 
                           root_dir='datasets/tiny-imagenet-200'):
        # if root_dir does not exist, download the dataset
        download = not os.path.exists(root_dir)

        train_dataset = TinyImageNetDataset(root_dir, mode='train', preload=False, download=download)
        self.train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False)
        self.img_size = (3, 64, 64)

        tiny_imagenet_paths = TinyImageNetPaths(root_dir, download=False)
        self.category_names = tiny_imagenet_paths.get_all_category_names()

        '''
        # get a list of all category names
        tiny_imagenet_paths = TinyImageNetPaths(root_dir, download=False)
        all_category_names = []
        for _, category_names in tiny_imagenet_paths.nid_to_words.items():
            all_category_names.extend(category_names)
        self.category_names = all_category_names
        '''
        # if somehow the above don't work to get the category names then we could also
        # use the info contained in the pre-trained ResNet50 model (at least that's known to
        # work) but it's not ideal as we don't want to have to load the model just to get the labels
        ''' # something like that
        _, _, self.img_size, self.category_names = load_data_aux(self.dataset_name, 
                                            data_dir=None, 
                                            layer_name=self.layer_name)
        _, self.weights = load_model_aux(self.model_name, 
                                         self.img_size, 
                                         expansion_factor=None)
        score, class_ids = self.classification_results_aux(output)

        category_names = [self.weights.meta["categories"][index] for index in class_ids]
        '''

    def load_cifar_10(self,
                      root_dir='datasets/cifar-10'):
        # if root_dir does not exist, download the dataset
        download = not os.path.exists(root_dir)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        ])

        # Data shuffling should be turned off here so that the activations rhat we store in the model without SAE
        # are in the same order as the activations that we store in the model with SAE
        train_dataset = torchvision.datasets.CIFAR10(root_dir, train=True, download=download, transform=transform)
        self.train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False)
        val_dataset = torchvision.datasets.CIFAR10(root_dir, train=False, download=download, transform=transform)
        self.val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        self.img_size = (3, 32, 32)
        self.category_names = train_dataset.classes
        # the classes are: ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        first_few_labels = [train_dataset.targets[i] for i in range(5)]
        print("Predefined labels:", first_few_labels)

    def load_intermediate_feature_maps(self,
                                       root_dir,
                                       layer_name):
        # NOT SURE ABOUT THE BELOW LINE, WANT TO BE ABLE TO ITERATE as such:
        # for inputs, targets in train_dataloader: and targets should be the same as inputs
        # because we want to compare the inputs with the inputs in the autoencoder
        dataset = IntermediateActivationsDataset(layer_name=layer_name, 
                                                 root_dir=root_dir, 
                                                 batch_size=self.batch_size)
        self.train_dataloader = DataLoader(dataset, 
                                    batch_size=self.batch_size, 
                                    shuffle=True)
        self.img_size = dataset.get_image_size()
        self.val_dataloader = None
        self.category_names = None

    def load_data(self, root_dir=None, layer_name=None):
        if self.dataset_name == 'tiny_imagenet':
            self.load_tiny_imagenet()
        elif self.dataset_name == 'cifar_10':
            self.load_cifar_10()
        elif self.dataset_name == 'intermediate_feature_maps':
            self.load_intermediate_feature_maps(root_dir, layer_name)
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")

'''
if __name__ == "__main__":
    data_loader = InputDataLoader(dataset_name)
    data_loader.load_data()

    # Access the loaded data
    loaded_data = data_loader.data

    # Further processing or model inference using loaded_data
    # ...
'''