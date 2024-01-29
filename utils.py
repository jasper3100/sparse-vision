import torch.nn.functional as F
import torch.nn as nn
import torch
import os
from torchvision.models import resnet50, ResNet50_Weights
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import h5py
import matplotlib.pyplot as plt
import numpy as np

from dataloaders.tiny_imagenet import TinyImageNetDataset, TinyImageNetPaths
from models.custom_mlp_1 import CustomMLP1
from models.sae_conv import SaeConv
from models.sae_mlp import SaeMLP
from losses.sparse_loss import SparseLoss

def get_optimizer(optimizer_name, model, learning_rate):
    if optimizer_name == 'adam':
        return torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=learning_rate) # momentum= 0.9, weight_decay=1e-4)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

def get_criterion(criterion_name, lambda_sparse=None):
    if criterion_name == 'cross_entropy':
        return nn.CrossEntropyLoss()
    elif criterion_name == 'sae_loss':
        return SparseLoss(lambda_sparse=lambda_sparse)
    else:
        raise ValueError(f"Unsupported criterion: {criterion_name}")
    
def get_img_size(dataset_name):
    if dataset_name == 'tiny_imagenet':
        return (3, 64, 64)
    elif dataset_name == 'cifar_10':
        return (3, 32, 32)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
def save_model_weights(model, folder_path, file_name='model_weights.pth', layer_name=None):
    os.makedirs(folder_path, exist_ok=True) # create folder if it doesn't exist

    # layer_name is used for SAE models, because SAE is trained on activations
    # of a specific layer
    if layer_name is not None:
        file_path = os.path.join(folder_path, f'{layer_name}_{file_name}')
    else:
        file_path = os.path.join(folder_path, file_name)
        
    torch.save(model.state_dict(), file_path)
    print(f"Successfully stored model weights in {file_path}")

def load_model(model_name, img_size=None, expansion_factor=None):
    if model_name == 'resnet50':
        return resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    elif model_name == 'custom_mlp_1':
        return CustomMLP1(img_size)
    elif model_name == 'sae_conv':
        return SaeConv(img_size, expansion_factor)
    elif model_name == 'sae_mlp':
        return SaeMLP(img_size, expansion_factor)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

def load_data(dataset_name, batch_size):
    if dataset_name == 'tiny_imagenet':
        root_dir='datasets/tiny-imagenet-200'
        # if root_dir does not exist, download the dataset
        download = not os.path.exists(root_dir)

        # Data shuffling should be turned off here so that the activations that we store in the model without SAE
        # are in the same order as the activations that we store in the model with SAE
        train_dataset = TinyImageNetDataset(root_dir, mode='train', preload=False, download=download)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        val_dataset = TinyImageNetDataset(root_dir, mode='val', preload=False, download=download)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        tiny_imagenet_paths = TinyImageNetPaths(root_dir, download=False)
        category_names = tiny_imagenet_paths.get_all_category_names()

        return train_dataloader, val_dataloader, category_names
    
    elif dataset_name == 'cifar_10':
        root_dir='datasets/cifar-10'
        download = not os.path.exists(root_dir)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        ])

        train_dataset = torchvision.datasets.CIFAR10(root_dir, train=True, download=download, transform=transform)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        val_dataset = torchvision.datasets.CIFAR10(root_dir, train=False, download=download, transform=transform)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        category_names = train_dataset.classes
        # the classes are: ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        #first_few_labels = [train_dataset.targets[i] for i in range(5)]
        #print("Predefined labels:", first_few_labels)

        train_dataset_length = len(train_dataset) # alternatively: train_dataset.__len__()
        # on the contrary: len(train_dataloader) returns the number of batches

        return train_dataloader, val_dataloader, category_names, train_dataset_length

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

def store_feature_maps(activations, folder_path):
    os.makedirs(folder_path, exist_ok=True) # create the folder

    # store the intermediate feature maps
    for name in activations.keys():
        activation = activations[name]

        # if activation is empty, we give error message
        if activation.nelement() == 0:
            raise ValueError(f"Activation of layer {name} is empty")
        else:
            activations_file_path = os.path.join(folder_path, f'{name}_activations.h5')
            # Store activations to an HDF5 file
            with h5py.File(activations_file_path, 'w') as h5_file:
                h5_file.create_dataset('data', data=activation.numpy())

def load_feature_map(file_path):
    with h5py.File(file_path, 'r') as h5_file:
        data = torch.from_numpy(h5_file['data'][:])
    # if data is emtpy, we give error message
    if data.nelement() == 0:
        raise ValueError(f"Trying to load feature map but it is empty")
    else:
        return data
    
def get_module_names(model):
    """
    Get the names of all named modules in the model.
    """
    layer_names = [name for name, _ in model.named_modules()]
    layer_names = list(filter(None, layer_names)) # remove emtpy strings
    return layer_names

def load_pretrained_model(model_name, 
                        img_size, 
                        weights_folder_path,
                        sae_expansion_factor=None, # only needed for SAE models
                        layer_name=None): # only needed for SAE models, 
                        # which are trained on activations of a specific layer
    model = load_model(model_name, img_size, sae_expansion_factor)
    if layer_name is not None:
        file_name = f'{layer_name}_model_weights.pth'
    else:
        file_name = 'model_weights.pth'
    weights_file_path = os.path.join(weights_folder_path, file_name)
    model.load_state_dict(torch.load(weights_file_path))
    model.eval()
    return model

def get_classifications(output, category_names=None):
    # if output is already a prob distribution (f.e. if last layer of network is softmax)
    # then we don't apply softmax. Applying softmax twice would be wrong.
    if torch.allclose(output.sum(dim=1), torch.tensor(1.0), atol=1e-3) and (output.min().item() >= 0) and (output.max().item() <= 1):
        prob = output
    else:
        prob = F.softmax(output, dim=1)
    scores, class_ids = prob.max(dim=1)
    if category_names is not None:
        category_list = [category_names[index] for index in class_ids]
    else:
        category_list = None
    return scores, category_list, class_ids

def show_classification_with_images(train_dataloader,
                                    class_names, 
                                    model=None,
                                    device=None,
                                    output=None,
                                    output_2=None):
    '''
    This function either works with available model output or the model can be used to generate the output.
    '''
    os.makedirs('evaluation_results', exist_ok=True)
    save_path = 'evaluation_results/classification_visualizations_original.png'
    
    n = 10  # show only the first n images, 
    # for showing all images in the batch use len(predicted_classes)

    input_images, target_ids = next(iter(train_dataloader))  
    input_images, target_ids = input_images[:n], target_ids[:n] # only show the first 10 images
    input_images, target_ids = input_images.to(device), target_ids.to(device)

    if model is not None:
        output = model(input_images)
    
    scores, predicted_classes, class_ids = get_classifications(output, class_names)
    if output_2 is not None:
        scores_2, predicted_classes_2, _ = get_classifications(output_2, class_names)
        save_path = 'evaluation_results/classification_visualizations_original_modified.png'
    
    fig, axes = plt.subplots(1, n + 1, figsize=(20, 3))

    # Add a title column to the left
    title_column = 'True\nOriginal Prediction\nModified Prediction'
    axes[0].text(0.5, 0.5, title_column, va='center', ha='center', fontsize=8, wrap=True)
    axes[0].axis('off')

    for i in range(n):
        img = input_images[i] / 2 + 0.5 # unnormalize the image
        npimg = img.numpy()
        axes[i + 1].imshow(np.transpose(npimg, (1, 2, 0)))

        if output_2 is not None:
            title = f'{class_names[target_ids[i]]}\n{predicted_classes[i]} ({100*scores[i].item():.1f}%)\n{predicted_classes_2[i]} ({100*scores_2[i].item():.1f}%)'
        else:
            title = f'{class_names[target_ids[i]]}\n{predicted_classes[i]} ({100*scores[i].item():.1f}%)'

        axes[i + 1].set_title(title, fontsize=8)
        axes[i + 1].axis('off')

    plt.subplots_adjust(wspace=0.5)  # Adjust space between images
    plt.savefig(save_path)
    plt.close()
    #plt.show()


def print_model_accuracy(model, device, train_dataloader):
    correct_predictions = 0
    total_samples = 0
    for input, target in train_dataloader:
        input, target = input.to(device), target.to(device)
        output = model(input)
        _, _, class_ids = get_classifications(output)
        correct_predictions += (class_ids == target).sum().item()
        total_samples += target.size(0)
    accuracy = correct_predictions / total_samples
    print(f'Train accuracy: {accuracy * 100:.2f}%')

    ''' # Alternative: seems to be equally fast
    num_batches = 1563 #int(get_stored_number(original_activations_folder_path, 'num_batches.txt'))
    all_targets = []
    all_outputs = []
    batch_idx = 0

    for input, target in train_dataloader:
        input, target = input.to(device), target.to(device)
        batch_idx += 1
        output = model(input)
        all_targets.append(target)
        all_outputs.append(output)
        if batch_idx == num_batches:
                break
    target = torch.cat(all_targets, dim=0)
    output = torch.cat(all_outputs, dim=0)
    #print(target.shape)
    accuracy = get_accuracy(output, target)
    print(f'Train accuracy: {accuracy * 100:.2f}%')
    '''

def save_number(x, folder_path, file_path):
    os.makedirs(folder_path, exist_ok=True)
    file_path = os.path.join(folder_path, file_path)
    with open(file_path, 'w') as f:
        f.write(str(x))

def get_stored_number(folder_path, file_name):
    file_path = os.path.join(folder_path, file_name)
    with open(file_path, 'r') as file:
        x = float(file.read())
    return x 

def measure_sparsity(x, threshold):
    """
    Measure the sparsity of a tensor x. Usually, the output of the SAE encoder
    went through a ReLU and thus x.abs() = x, but we apply the absolute value here
    regardless for cases where no ReLU was applied. To compute the sparsity one has to
    do: 1 - (number of activating units / total number of units)
    """
    return (x.abs() > threshold).sum().item(), x.nelement()

def get_accuracy(output, target):
    _, _, class_ids = get_classifications(output)
    return (class_ids == target).sum().item() / target.size(0)