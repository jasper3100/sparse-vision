import torch
import torchvision
import torchvision.transforms as transforms
import os
from torch.utils.data import DataLoader

from tiny_imagenet import TinyImageNetDataset
from model_loader import ModelLoader

class InputDataLoader:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.data = None
        self.img_size = None

    def load_sample_data_1(self):
        torch.manual_seed(0)
        self.data = torch.randn(10, 3, 224, 224)
        self.img_size = (3, 224, 224)

    def load_image_data(self, 
                        image_path=r"C:\Users\Jasper\Downloads\Master thesis\Code\fox.jpg", 
                        model_name='resnet50'):
        img = torchvision.io.read_image(image_path)
        model = ModelLoader(model_name).load_model().model
        weights = model.weights
        preprocess = weights.transforms()
        self.data = preprocess(img).unsqueeze(0)
        self.img_size = (3, 224, 224)

    def load_tiny_imagenet(self, 
                           root_dir='datasets/tiny-imagenet-200',
                           batch_size=32):
        # if root_dir does not exist, download the dataset
        download = not os.path.exists(root_dir)

        train_dataset = TinyImageNetDataset(root_dir, mode='train', preload=False, download=download)
        self.data = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        self.img_size = (3, 64, 64)

    def load_cifar_10(self,
                      root_dir='datasets/cifar-10',
                      batch_size=32):
        # if root_dir does not exist, download the dataset
        download = not os.path.exists(root_dir)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        ])

        train_dataset = torchvision.datasets.CIFAR10(root_dir, train=True, download=download, transform=transform)
        self.data = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.img_size = (3, 32, 32)

    def load_data(self):
        if self.dataset_name == 'sample_data_1':
            self.load_sample_data_1()
        elif self.dataset_name == 'img':
            self.load_image_data()
        elif self.dataset_name == 'tiny_imagenet':
            self.load_tiny_imagenet()
        elif self.dataset_name == 'cifar_10':
            self.load_cifar_10()
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