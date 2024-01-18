import torch
import torchvision
import os
from torch.utils.data import DataLoader

from main import dataset_name
from model import weights
from tiny_imagenet import TinyImageNetDataset

'''
Define input data for the model.
'''

if dataset_name == 'sample_data_1':
    torch.manual_seed(0) # fix seed for reproducibility
    input_data = torch.randn(10, 3, 224, 224)  # shape: (batch_size, channels, height, width)

if dataset_name == 'img':
    img = torchvision.io.read_image(r"C:\Users\Jasper\Downloads\Master thesis\Code\fox.jpg")
    #("/mnt/qb/work/akata/aoq918/interpretability/fox.jpg")

    # Initialize the inference transforms
    preprocess = weights.transforms()

    # Apply inference preprocessing transforms
    input_data = preprocess(img).unsqueeze(0)
    #print(input_data.shape) # print input dimension of model

if dataset_name == 'tiny_imagenet':
    root_dir = 'datasets/tiny-imagenet-200'

    # if root_dir does not exist, download the dataset
    if not os.path.exists(root_dir):
        download=True
    else:
        download=False

    train_dataset = TinyImageNetDataset(root_dir, mode='train', preload=False, download=download)

    batch_size = 32
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)