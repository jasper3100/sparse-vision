import torch
import torchvision
from datasets import load_dataset

from main import dataset_name
from model import weights

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

#if dataset_name == 'tiny_imagenet':
def example_usage():
    tiny_imagenet = load_dataset('Maysee/tiny-imagenet', split='train')
    print(tiny_imagenet[0])

if __name__ == '__main__':
    example_usage()