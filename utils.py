import torch.nn.functional as F
import torch.nn as nn
import torch
import os
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import resnet18, ResNet18_Weights
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import h5py
import numpy as np
import wandb
import csv
import math
import matplotlib.pyplot as plt
from lucent.optvis import render
import lucent.optvis.param as lucentparam
from lucent.modelzoo.util import get_model_layers
from einops import rearrange

from dataloaders.tiny_imagenet import *
from dataloaders.tiny_imagenet import _add_channels
from models.custom_mlp import *
from models.custom_cnn import *
from models.sae_conv import SaeConv
from models.sae_mlp import SaeMLP
from losses.sparse_loss import SparseLoss
from supplementary.dataset_stats import print_dataset_stats
from dataloaders.imagenet import *

def get_optimizer(optimizer_name, model, learning_rate):
    if optimizer_name == 'adam':
        return torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0,0.9999)), None
    elif optimizer_name == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=learning_rate), None
    elif optimizer_name == 'sgd_w_scheduler':
        # here we also use momentum (for ResNet-18)
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)    
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        return optimizer, scheduler
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
    elif dataset_name == 'mnist':   
        return (1, 28, 28)
    elif dataset_name == 'imagenet':
        return (3, 224, 224)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
def get_file_path(folder_path=None, layer_names=None, params=None, file_name=None, params2=None):
    '''
    params and params2 expect a dictionary of parameters
    '''
    if folder_path is not None:
        os.makedirs(folder_path, exist_ok=True) # create the folder

    if params is not None:
        if isinstance(params, dict):
            values_as_strings = [str(value) if value is not None else "None" for value in params.values()]
            params = "_".join(values_as_strings)
        
        if params2 is not None:
            if isinstance(params2, dict):
                values_as_strings = [str(value) if value is not None else "None" for value in params2.values()]
                params2 = "_".join(values_as_strings)
            file_name = f'{layer_names}_{params}_{params2}_{file_name}'
        else:
            file_name = f'{layer_names}_{params}_{file_name}'
    else:
        file_name = f'{layer_names}_{file_name}'

    if folder_path is None:
        file_path = file_name
    else:
        file_path = os.path.join(folder_path, file_name)
    return file_path
    
def save_model_weights(model, 
                       folder_path, 
                       layer_names=None, # layer_name is used for SAE models, because SAE is trained on activations of a specific layer
                       params=None):
    os.makedirs(folder_path, exist_ok=True) # create folder if it doesn't exist
    file_path = get_file_path(folder_path, layer_names, params, 'model_weights.pth')
    torch.save(model.state_dict(), file_path)
    print(f"Successfully stored model weights in {file_path}")

def load_pretrained_model(model_name, 
                        img_size, 
                        folder_path,
                        num_classes=None,
                        sae_expansion_factor=None, # only needed for SAE models
                        layer_name=None,# only needed for SAE models, which are trained on activations of a specific layer
                        params=None,
                        execution_location=None):  
    model = load_model(model_name, img_size=img_size, num_classes=num_classes, expansion_factor=sae_expansion_factor)
    if model_name != "resnet18" and model_name != "resnet50":
        # we load the pre-trained resnet models in load_model directly	
        file_path = get_file_path(folder_path, layer_name, params, 'model_weights.pth')
        state_dict = torch.load(file_path)
        if "W_enc" in state_dict:
            state_dict["encoder.weight"] = state_dict.pop("W_enc")
            # take the transpose of the weight matrix
            state_dict["encoder.weight"] = state_dict["encoder.weight"].T
        if "b_enc" in state_dict:
            state_dict["encoder.bias"] = state_dict.pop("b_enc")
        if "W_dec" in state_dict:
            state_dict["decoder.weight"] = state_dict.pop("W_dec")
            # take the transpose of the weight matrix
            state_dict["decoder.weight"] = state_dict["decoder.weight"].T
        if "b_dec" in state_dict:
            state_dict["decoder.bias"] = state_dict.pop("b_dec")
        model.load_state_dict(state_dict)
        #model.load_state_dict(torch.load(file_path))
    model.eval()
    return model

def load_model(model_name, img_size=None, num_classes=None, expansion_factor=None, execution_location=None):
    if model_name == 'resnet50':
        return resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    elif model_name == 'resnet18_1':
        # FIRST ROUND OF FINE-TUNING
        # I follow this page on using ResNet18 with pre-trained for Imagenet but then
        # finetuning the model to work on Tiny Imagenet: https://github.com/tjmoon0104/Tiny-ImageNet-Classifier 
        model_ft = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        #Adjust final layers to tiny imagenet (200 instead of 1000 classes)
        model_ft.avgpool = nn.AdaptiveAvgPool2d(1)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, 200)
        return model_ft
    
    elif model_name == 'resnet18_2':
        # SECOND ROUND OF FINE-TUNING
        path = "/lustre/home/jtoussaint/master_thesis/model_weights/resnet18_1/tiny_imagenet/None_resnet18_1_4_0.001_100_sgd_w_scheduler_0.1_model_weights.pth"
        model_ft = resnet18()   
        # Adjust final layers as before 
        model_ft.avgpool = nn.AdaptiveAvgPool2d(1)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, 200) 

        model_ft.load_state_dict(torch.load(path))

        # Adjust layers at beginning
        model_ft.conv1 = nn.Conv2d(3, 64, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        model_ft.maxpool = nn.Sequential() # nn.Identity() # remove maxpool layer

        return model_ft
    
    elif model_name == 'resnet18':
        # PRE-TRAINED MODEL
        model_ft = resnet18()
        # Adjust final layers 
        model_ft.avgpool = nn.AdaptiveAvgPool2d(1)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, 200) 
        # Adjust layers at beginning
        model_ft.conv1 = nn.Conv2d(3, 64, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        model_ft.maxpool = nn.Sequential() # nn.Identity() # remove maxpool layer

        path = "/lustre/home/jtoussaint/master_thesis/model_weights/resnet18_2/tiny_imagenet/None_resnet18_2_10_0.001_100_sgd_w_scheduler_0.1_model_weights.pth"
        # if this path exists, load the weights
        if os.path.exists(path):
            model_ft.load_state_dict(torch.load(path))
        # else, if we are on local machine, just use random weights

        return model_ft
    
    elif model_name == 'custom_mlp_1':
        return CustomMLP1(img_size, num_classes)
    elif model_name == 'custom_mlp_2':
        return CustomMLP2(img_size, num_classes)
    elif model_name == 'custom_mlp_3':
        return CustomMLP3(img_size, num_classes)
    elif model_name == 'custom_mlp_4':
        return CustomMLP4(img_size, num_classes)
    elif model_name == 'custom_mlp_5':
        return CustomMLP5(img_size, num_classes)
    elif model_name == 'custom_mlp_6':
        return CustomMLP6(img_size, num_classes)
    elif model_name == "custom_mlp_7":
        return CustomMLP7(img_size, num_classes)
    elif model_name == "custom_mlp_8":
        return CustomMLP8(img_size, num_classes)
    elif model_name == 'custom_mlp_9':
        return CustomMLP9(img_size, num_classes)
    elif model_name == 'custom_mlp_10':
        return CustomMLP10(img_size, num_classes)
    elif model_name == 'custom_cnn_1':
        return CustomCNN1(img_size, num_classes)
    elif model_name == 'sae_conv':
        return SaeConv(img_size, expansion_factor)
    elif model_name == 'sae_mlp':
        return SaeMLP(img_size, expansion_factor)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

def load_data(directory_path, dataset_name, batch_size):
    # shuffling the data during training has some advantages, during evaluation it's not necessary
    train_shuffle=True
    eval_shuffle=False
    # we drop the last batch if it has less than batch_size samples (which might happen if the 
    # dataset size is not divisible by batch_size). This ensures that when we average quantities over
    # batches, we don't have to worry about the last batch being smaller than the others.
    drop_last=True
    if dataset_name == 'tiny_imagenet':
        if directory_path.startswith('/lustre'):
            #root_dir='/lustre/home/jtoussaint/master_thesis/datasets/tiny-imagenet-200'
            root_dir = os.path.join(directory_path, 'datasets/tiny-imagenet-200')
        elif directory_path.startswith('C:'):
            root_dir = os.path.join(directory_path, 'datasets\\tiny-imagenet-200')
        else:
            raise ValueError(f"Unexpected directory path {directory_path}")
        # if root_dir does not exist, download the dataset
        download = not os.path.exists(root_dir)
        if not os.path.exists(root_dir):
            os.makedirs(root_dir, exist_ok=True)

        # normalization of the images is performed within the TinyImageNetDataset class
        train_dataset = TinyImageNetDataset(root_dir, mode='train', preload=False, download=download)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=train_shuffle, drop_last=drop_last, num_workers=5)
        val_dataset = TinyImageNetDataset(root_dir, mode='val', preload=False, download=download)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=eval_shuffle, drop_last=drop_last, num_workers=5)

        tiny_imagenet_paths = TinyImageNetPaths(root_dir, download=False)
        category_names = tiny_imagenet_paths.get_all_category_names()
    
    elif dataset_name == 'cifar_10':
        root_dir='datasets/cifar-10'
        # if root_dir does not exist, download the dataset
        root_dir=os.path.join(directory_path, root_dir)
        # The below works on the cluster
        #root_dir='/lustre/home/jtoussaint/master_thesis/datasets/cifar-10'
        download = not os.path.exists(root_dir)
        if not os.path.exists(root_dir):
            os.makedirs(root_dir, exist_ok=True)

        #'''
        data_resolution = 32
        mean = [0.4913997551666284, 0.48215855929893703, 0.4465309133731618]
        # has to be tensor
        data_mean = torch.tensor(mean)
        # has to be tuple
        data_mean_int = []
        for c in range(data_mean.numel()):
            data_mean_int.append(int(255 * data_mean[c]))
        data_mean_int = tuple(data_mean_int)
        train_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
        # transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
        '''
            #transforms.ToTensor()
            # Cifar images have values in [0,1]. We scale them to be in [-1,1].
            # Some thoughts on why [-1,1] is better than [0,1]:
            # datascience.stackexchange.com/questions/54296/should-input-images-be-normalized-to-1-to-1-or-0-to-1
            # We can verify that using the values (0.5,0.5,0.5),(0.5,0.5,0.5) will yield data in [-1,1]:
            # verify using: print_dataset_stats(train_dataset)
            # to call this function, in this script, do:
            # load_data(directory_path=directory_path, dataset_name='cifar_10', batch_size=32)
            # Theoretical explanation: [0,1] --> (0 - 0.5)/0.5 = -1 and (1 - 0.5)/0.5 = 1
            # instead, using the mean and std instead, will yield normalized data (mean=0, std=1) but not in [-1,1]
            #transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))

            #torchvision.transforms.ToPILImage(),
            #torchvision.transforms.RandomCrop(data_resolution, padding=int(data_resolution * 0.125), fill=data_mean_int),
            #torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.AutoAugment(policy=torchvision.transforms.autoaugment.AutoAugmentPolicy.CIFAR10),#, fill=data_mean_int),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
            #common.autoaugment.CutoutAfterToTensor(n_holes=1, length=cutout, fill_color=data_mean),
            #torchvision.transforms.RandomErasing(value=data_mean_int),
        ])
        '''
        test_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
        #transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

        # dataloader argument num_workers = 2 ? 
        train_dataset = torchvision.datasets.CIFAR10(root_dir, train=True, download=download, transform=train_transform)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=train_shuffle, drop_last=drop_last)
        val_dataset = torchvision.datasets.CIFAR10(root_dir, train=False, download=download, transform=test_transform)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=eval_shuffle, drop_last=drop_last)
            
        category_names = train_dataset.classes
        # the classes are: ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        #first_few_labels = [train_dataset.targets[i] for i in range(5)]
        #print("Predefined labels:", first_few_labels)
    
    elif dataset_name == 'mnist':
        root_dir='datasets/mnist'
        # if root_dir does not exist, download the dataset
        root_dir=os.path.join(directory_path, root_dir)
        download = not os.path.exists(root_dir)
        if not os.path.exists(root_dir):
            os.makedirs(root_dir, exist_ok=True)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        train_dataset = torchvision.datasets.MNIST(root_dir, train=True, download=download, transform=transform)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=train_shuffle, drop_last=drop_last)
        val_dataset = torchvision.datasets.MNIST(root_dir, train=False, download=download, transform=transform)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=eval_shuffle, drop_last=drop_last)
        
        category_names = train_dataset.classes

    elif dataset_name == 'imagenet':    
        '''         
        # Define transformations to be applied to the images
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        '''

        # THIS CODE IN THIS VERSION EXCEEDS THE CURRENT MEMORY LIMITS THAT I SET IN THE CLUSTER. WHY?

        root_dir = '/lustre/shared/imagenet-2016/ILSVRC'

        # Define the dataset for training, validation, and testing
        #train_dataset = ImageNetDataset(root_dir=root_dir, split='train')#, transform=transform)
        val_dataset = ImageNetDataset(root_dir=root_dir, split='val')#, transform=transform)
        #test_dataset = ImageNetDataset(root_dir=root_dir, split='test')#, transform=transform)

        # Create DataLoader for training, validation, and testing
        #train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
        val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)
        #test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

        category_names = train_dataset.classes

        '''
        root_dir='/lustre/shared/imagenet-2016'

        #transform = transforms.Compose([
        #    transforms.ToTensor(),
        #    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        #])

        train_dataset = torchvision.datasets.ImageNet(root_dir, split='train', download=False)#, transform=transform)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=train_shuffle, drop_last=drop_last)
        val_dataset = torchvision.datasets.ImageNet(root_dir, split='val', download=False)#, transform=transform)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=eval_shuffle, drop_last=drop_last)
        
        category_names = train_dataset.classes
        '''
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    img_size = get_img_size(dataset_name)

    return train_dataloader, val_dataloader, category_names, img_size

    
def store_feature_maps(activations, folder_path, params=None):
    os.makedirs(folder_path, exist_ok=True) # create the folder

    # store the intermediate feature maps
    for name in activations.keys():
        activation = activations[name]

        # if activation is empty, we give error message
        if activation.nelement() == 0:
            raise ValueError(f"Activation of layer {name} is empty")
        else:
            file_path = get_file_path(folder_path, layer_names=[name], params=params, file_name='activations.h5')
            # Store activations to an HDF5 file
            with h5py.File(file_path, 'w') as h5_file:
                h5_file.create_dataset('data', data=activation.cpu().numpy())

def store_batch_feature_maps(activation, num_samples, name, folder_path, params=None):
    os.makedirs(folder_path, exist_ok=True) # create the folder

    # if activation is empty, we give error message
    if activation.nelement() == 0:
        raise ValueError(f"Activation of layer {name} is empty")
    else:
        file_path = get_file_path(folder_path, layer_names=[name], params=params, file_name='activations.h5')
        # Store activations to an HDF5 file
        with h5py.File(file_path, 'a') as h5_file:
            # check if the dataset already exists
            if 'data' not in h5_file:
                # initialize the dataset with a predefined size
                h5_file.create_dataset('data', shape=(num_samples,) + activation.shape[1:])#, dtype=activation.dtype)
                next_position = 0
            else:
                # determine the next available position
                next_position = h5_file['data'].shape[0]

            h5_file['data'][next_position:next_position + activation.shape[0]] = activation.cpu().numpy()

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

def show_classification_with_images(dataloader,
                                    class_names, 
                                    wandb_status,
                                    folder_path=None,
                                    layer_names=None,
                                    model=None,
                                    device=None,
                                    output=None,
                                    output_2=None,
                                    params=None):
    '''
    This function either works with available model output or the model can be used to generate the output.
    '''
    os.makedirs(folder_path, exist_ok=True)
    file_path = get_file_path(folder_path, layer_names, params, 'classif_visual_original.png')
    
    n = 10  # show only the first n images, 
    # for showing all images in the batch use len(predicted_classes)

    for batch in dataloader:
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            input_images, target_ids = batch
        elif isinstance(batch, dict) and len(batch) == 2 and list(batch.keys())[0] == "image" and list(batch.keys())[1] == "label":
            # this format holds for the tiny imagenet dataset
            input_images, target_ids = batch["image"], batch["label"]
        else:
            raise ValueError("Unexpected data format from dataloader")
        break # we only need the first batch

    input_images, target_ids = input_images[:n], target_ids[:n] # only show the first 10 images
    input_images, target_ids = input_images.to(device), target_ids.to(device)

    if model is not None:
        output = model(input_images)
        output = F.softmax(output, dim=1)
    
    scores, predicted_classes, _ = get_classifications(output, class_names)
    if output_2 is not None:
        scores_2, predicted_classes_2, _ = get_classifications(output_2, class_names)
        file_path = get_file_path(folder_path, layer_names, params, 'classif_visual_original_modified.png')
    
    fig, axes = plt.subplots(1, n + 1, figsize=(20, 3))

    # Add a title column to the left
    title_column = 'True\nOriginal Prediction\nModified Prediction'
    axes[0].text(0.5, 0.5, title_column, va='center', ha='center', fontsize=8, wrap=True)
    axes[0].axis('off')

    for i in range(n):
        if "tiny_imagenet" in folder_path:
            # input_images[i].shape) --> torch.Size([3, 64, 64])
            img = input_images[i]
            #img = _add_channels(img) 
            # we don't need to unnormalize the image
            img = img.cpu().numpy()
            img = img.astype(int)
            axes[i + 1].imshow(np.transpose(img, (1, 2, 0)))
        elif "mnist" in folder_path:
            #mean=(0.1307,)
            #std=(0.3081,)
            #img = input_images[i] * np.array(std) + np.array(mean)
            #img = np.clip(img, 0, 1)
            img = input_images[i].cpu().numpy()
            axes[i+1].imshow(img.squeeze(), cmap='gray')
        else:
            # for all other datasets, the following seems to work
            img = input_images[i] / 2 + 0.5 # unnormalize the image
            img = img.cpu().numpy()
            axes[i + 1].imshow(np.transpose(img, (1, 2, 0)))

        if output_2 is not None:
            title = f'{class_names[target_ids[i]]}\n{predicted_classes[i]} ({100*scores[i].item():.1f}%)\n{predicted_classes_2[i]} ({100*scores_2[i].item():.1f}%)'
        else:
            title = f'{class_names[target_ids[i]]}\n{predicted_classes[i]} ({100*scores[i].item():.1f}%)'

        axes[i + 1].set_title(title, fontsize=8)
        axes[i + 1].axis('off')

    plt.subplots_adjust(wspace=0.5) # Adjust space between images
    
    if wandb_status:
        wandb.log({f"eval/classification_visualization":wandb.Image(plt)})
    else:
        plt.savefig(file_path)
        plt.close()
        #plt.show()
        print(f"Successfully stored classification visualizations in {file_path}")

def log_image_table(dataloader,
                    class_names, 
                    model=None,
                    device=None,
                    output=None,
                    output_2=None):
    n = 10  # show only the first n images, 
    # for showing all images in the batch use len(predicted_classes)

    images, target_ids = next(iter(dataloader))  
    images, target_ids = images[:n], target_ids[:n] # only show the first 10 images
    images, target_ids = images.to(device), target_ids.to(device)

    if model is not None:
        output = model(images)
        output = F.softmax(output, dim=1)
    
    scores, predicted_classes, _ = get_classifications(output, class_names)
    if output_2 is not None:
        scores_2, predicted_classes_2, _ = get_classifications(output_2, class_names)
        table = wandb.Table(columns=["image", "target", "pred","scores","pred_2","scores_2"])
        for image, predicted_class, target_id, score, predicted_class_2, score_2 in zip(images,
                                                                                        predicted_classes, 
                                                                                        target_ids,#.to("cpu"), 
                                                                                        scores, 
                                                                                        predicted_classes_2, 
                                                                                        scores_2):
            img = image / 2 + 0.5 # moves data from [-1,1] to [0,1] i.e. we assume here that original data was in [0,1]
            img = np.transpose(img.cpu().numpy(), (1,2,0)) # WHAT EXACTLY IS THIS LINE DOING??? CHECK IF IT MAKES SENSE FOR ALL DATASETS...
            score = 100*score.item()
            score_2 = 100*score_2.item()
            table.add_data(wandb.Image(img),
                            class_names[target_id],
                            predicted_class,
                            score,
                            predicted_class_2,
                            score_2)
        wandb.log({"original_and_modified_predictions_table":table}, commit=False)
    else:
        table = wandb.Table(columns=["image", "target", "pred", "scores"])
        for image, predicted_class, target_id, score in zip(images, 
                                                            predicted_classes, 
                                                            target_ids,#.to("cpu"), 
                                                            scores):#.to("cpu")):
            img = image / 2 + 0.5 # unnormalize the image
            img = np.transpose(img.cpu().numpy(), (1,2,0))
            score = 100*score.item()
            table.add_data(wandb.Image(img), 
                            class_names[target_id], 
                            predicted_class, 
                            score)#*score.detach().numpy())
        wandb.log({"original_predictions_table":table}, commit=False)

def plot_active_classes_per_neuron(number_active_classes_per_neuron, 
                                   layer_names, 
                                   num_classes,
                                   folder_path=None, 
                                   params=None, 
                                   wandb_status=None):
    # for now, only show results for the specified layer and the last layer, which is 'fc3'
    # 2 rows for the 2 layers we're considering, and 3 columns for the 3 types of models (original, modified, sae)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Number of active classes per neuron')
    bins = np.arange(0, num_classes+1.01, 1)

    for name in number_active_classes_per_neuron.keys():
        if name[0] == layer_names.split("_")[-1] or name[0] == 'fc3':
            if name[0] == 'fc3':
                row = 1
            else:
                row = 0
            if name[1] == 'original':
                col = 0
            elif name[1] == 'modified':
                col = 1
            elif name[1] == 'sae':
                col = 2
            axes[row, col].hist(number_active_classes_per_neuron[name].cpu().numpy(), bins=bins, color='blue', edgecolor='black')
            axes[row, col].set_title(f'Layer: {name[0]}, Model: {name[1]}')
            axes[row, col].set_xlabel('Number of active classes')
            axes[row, col].set_ylabel('Number of neurons')
            axes[row, col].set_xticks(np.arange(0.5, num_classes+1, 1))
            axes[row, col].set_xticklabels([str(int(x)) for x in range(num_classes+1)])

    plt.subplots_adjust(wspace=0.7)  # Adjust space between images
    
    if wandb_status:
        wandb.log({"active_classes_per_neuron":wandb.Image(plt)})
    else:
        os.makedirs(folder_path, exist_ok=True)
        file_path = get_file_path(folder_path, layer_names, params, 'active_classes_per_neuron.png')
        plt.savefig(file_path)
        plt.close()
        print(f"Successfully stored active classes per neuron plot in {file_path}")

def plot_neuron_activation_density(active_classes_per_neuron, 
                                   layer_names, 
                                   num_samples,
                                   folder_path=None, 
                                   params=None, 
                                   wandb_status=None):
    # for now, only show results for the specified layer and the last layer, which is 'fc3'
    # 2 rows for the 2 layers we're considering, and 3 columns for the 3 types of models (original, modified, sae)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Neuron activation density')
    bins = np.arange(0, 1.01, 0.05)

    for name in active_classes_per_neuron.keys():
        if name[0] == layer_names.split("_")[-1] or name[0] == 'fc3':
            if name[0] == 'fc3':
                row = 1
            else:
                row = 0
            if name[1] == 'original':
                col = 0
            elif name[1] == 'modified':
                col = 1
            elif name[1] == 'sae':
                col = 2
            
            # For each row we sum up the entries in that row --> how often each neuron was active overall
            how_often_neuron_was_active = torch.sum(active_classes_per_neuron[name], dim=1)
            # active_classes_per_neuron is stored epoch-wise, thus we can normalize by the number of samples in one epoch
            how_often_neuron_was_active = how_often_neuron_was_active / num_samples

            axes[row, col].hist(how_often_neuron_was_active.cpu().numpy(), bins=bins, color='green', edgecolor='black')
            axes[row, col].set_title(f'Layer: {name[0]}, Model: {name[1]}')
            axes[row, col].set_xlabel('Percentage of samples on which a neuron is active')
            axes[row, col].set_ylabel('Number of neurons')
 
    plt.subplots_adjust(wspace=0.7)  # Adjust space between images
    
    if wandb_status:
        wandb.log({"neuron_activation_density":wandb.Image(plt)})
    else:
        os.makedirs(folder_path, exist_ok=True)
        file_path = get_file_path(folder_path, layer_names, params, 'neuron_activation_density.png')
        plt.savefig(file_path)
        plt.close()
        print(f"Successfully stored neuron activation density plot in {file_path}")


def save_numbers(numbers, file_path):
    # input can be a tuple or a list of numbers
    with open(file_path, 'w') as f:
        # Convert to a comma-separated string
        numbers_str = ','.join(map(str, numbers))
        f.write(numbers_str)

def get_stored_numbers(file_path):
    with open(file_path, 'r') as file:
        # Read the string from the file and split it into a list of strings
        numbers_str = file.read()
        # Convert the list of strings to a list of floats
        numbers = list(map(float, numbers_str.split(',')))
    return numbers

def measure_activating_neurons(x, threshold):
    """
    Measure the number of activating neurons.
    """
    x = x.abs()
    inactive_neurons = x < threshold # get the indices of the neurons below the threshold
    
    inactive_neurons = torch.prod(inactive_neurons, dim=0) # we collapse the batch size dimension. In particular,
    # inactive_neurons is of shape [batch size (samples in batch), number of neurons in layer]. We multiply the 
    # rows element-wise, so that we get a tensor of shape [number of neurons in layer], where each
    # element is True if the neuron is dead in all batches, and False otherwise.

    # the below quantity is summed over all samples in one batch
    number_active_neurons = (x >= threshold).sum().item() / x.shape[0] # divide by batch size to get the average

    # get the total number of neurons in the layer, i.e. the second dimension of x
    number_total_neurons = x.shape[1]

    return inactive_neurons, number_active_neurons, number_total_neurons

def calculate_accuracy(output, target):
    _, _, class_ids = get_classifications(output)
    return (class_ids == target).sum().item() / target.size(0)

def get_target_output(device, train_dataloader, model=None, num_batches=None):
    '''
    If no model is provided, then only target is returned; otherwise target, output
    '''
    all_targets = []
    all_outputs = []
    batch_idx = 0
    for input, target in train_dataloader:
        input, target = input.to(device), target.to(device)
        batch_idx += 1
        if model is not None:
            output = model(input)
            all_outputs.append(output)
        all_targets.append(target)
        if batch_idx == num_batches:
            break
    target = torch.cat(all_targets, dim=0)

    if model is not None:
        output = torch.cat(all_outputs, dim=0)
        return target, output
    else:
        return target

def get_model_accuracy(model, 
                       device, 
                       train_dataloader, 
                       original_activations_folder_path=None, 
                       layer_names=None, 
                       model_params=None, 
                       wandb_status=False,
                       num_batches=None):
    target, output = get_target_output(device,train_dataloader,model=model,num_batches=num_batches)
    output = F.softmax(output, dim=1)
    accuracy = calculate_accuracy(output, target)
    print(f'Train accuracy: {accuracy * 100:.2f}%')
    if wandb_status:
        wandb.log({"model_train_accuracy": accuracy})

def store_sae_eval_results(folder_path, 
                            layer_names, 
                            params, 
                            lambda_sparse, 
                            expansion_factor, 
                            rec_loss, 
                            scaled_l1_loss, 
                            nrmse_loss,
                            rmse_loss,
                            sparsity,
                            sparsity_1):
    # remove lambda_sparse and expansion_factor from params, because we want a uniform file name 
    # for all lambda_sparse and expansion_factor values
    file_path = get_file_path(folder_path=folder_path,
                            layer_names=layer_names,
                            params=params,
                            file_name='sae_eval_results.csv')
    file_exists = os.path.exists(file_path)
    columns = ["lambda_sparse", "expansion_factor", "rec_loss", "l1_loss", "nrmse_loss", "rmse_loss", "rel_sparsity", "rel_sparsity_1"]
        
    if not file_exists:
        with open(file_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=columns)
            writer.writeheader()
            writer.writerow({columns[0]: lambda_sparse,
                            columns[1]: expansion_factor,
                            columns[2]: rec_loss,
                            columns[3]: scaled_l1_loss,
                            columns[4]: nrmse_loss,
                            columns[5]: rmse_loss,
                            columns[6]: sparsity,
                            columns[7]: sparsity_1})
    else:
        # Read the existing CSV file
        with open(file_path, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile, fieldnames=columns)
            rows = list(reader)

            # Check if the combination of lambda_sparse, expansion_factor already exists
            combination_exists = any(row["lambda_sparse"] == str(lambda_sparse) and row["expansion_factor"] == str(expansion_factor) for row in rows)

            # If the combination exists, update rec_loss, l1_loss, relative sparsity
            if combination_exists:
                for row in rows:
                    if row["lambda_sparse"] == str(lambda_sparse) and row["expansion_factor"] == str(expansion_factor):
                        row["rec_loss"] = str(rec_loss)
                        row["l1_loss"] = str(scaled_l1_loss)
                        row["nrmse_loss"] = str(nrmse_loss)
                        row["rmse_loss"] = str(rmse_loss)
                        row["rel_sparsity"] = str(sparsity)
                        if sparsity_1 is not None:
                            row["rel_sparsity_1"] = str(sparsity_1)
                        break
            else:
                # If the combination doesn't exist, add a new row
                rows.append({"lambda_sparse": str(lambda_sparse), 
                             "expansion_factor": str(expansion_factor), 
                             "rec_loss": str(rec_loss), 
                             "l1_loss": str(scaled_l1_loss),
                             "nrmse_loss": str(nrmse_loss),
                             "rmse_loss": str(rmse_loss),
                             "rel_sparsity": str(sparsity),
                             "rel_sparsity_1": str(sparsity_1)})

        # Write the updated data back to the CSV file
        with open(file_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=columns)
            writer.writerows(rows)
    print(f"Successfully stored SAE eval results with lamda_sparse and expansion_factor in {file_path}")

def get_folder_paths(directory_path, model_name, dataset_name, sae_model_name):
    model_weights_folder_path = os.path.join(directory_path, 'model_weights', model_name, dataset_name)
    sae_weights_folder_path = os.path.join(directory_path, 'model_weights', sae_model_name, dataset_name)
    #activations_folder_path = os.path.join(directory_path, 'feature_maps', model_name, dataset_name)
    evaluation_results_folder_path = os.path.join(directory_path, 'evaluation_results', model_name, dataset_name)
    return model_weights_folder_path, sae_weights_folder_path, evaluation_results_folder_path


def active_classes_per_neuron_aux(encoder_output, target, num_classes, activation_threshold):
    '''
    Return a matrix showing for each neuron and each class, how often the neuron was active on a sample
    of that class over one batch. 
    '''
    # targets is of the form [6,9,9,1,2,...], i.e., it contains the target class indices
    # target shape [number of samples n]
    #n = target.shape[0]
    # encoder_output.shape [number of samples n, number of neurons in augmented layer d]
    d = encoder_output.shape[1]

    # Create a binary mask for values above the activation_threshold
    above_threshold = encoder_output >= activation_threshold
    # We create a matrix of size [row = d, column = num_classes] where each row i, contains for a certain
    # dimension i of all activations, the number of times a class j has an activation
    # above the threshold.
    #print(d, num_classes)
    counting_matrix = torch.zeros(d, num_classes)
    above_threshold = above_threshold.to(counting_matrix.dtype)  # Convert to the same type
    counting_matrix.index_add_(1, target, above_threshold.t())
    # The code is equivalent to the below for loop, which is too slow though but easier to understand
    '''
    for i in range(n):
        for j in range(d):
            counting_matrix[j, target[i]] += encoder_output[i, j] >= activation_threshold
    '''
    return counting_matrix

def compute_number_dead_neurons(dead_neurons):
    '''
    dead_neurons is a dictionary of the form {(model_type,layer_name): [True,False,False,...]},
    where 'True' means that the neuron is dead
    Hence, for each key in dead_neurons, count the number of "True"'s
    '''
    number_dead_neurons = {}
    for key, tensor in dead_neurons.items():
        number_dead_neurons[key] = tensor.sum().item()
    return number_dead_neurons

def print_and_log_results(train_or_eval,
                          model_loss,
                          accuracy,
                          use_sae,
                          wandb_status,
                          sparsity_dict=None,
                          mean_number_active_classes_per_neuron=None,
                          std_number_active_classes_per_neuron=None,
                          number_active_neurons=None,
                          sparsity_dict_1=None,
                          number_dead_neurons=None,
                          batch=None,
                          epoch=None,
                          sae_loss=None,
                          sae_rec_loss=None,
                          sae_l1_loss=None,
                          sae_nrmse_loss=None,
                          sae_rmse_loss=None,
                          kld=None,
                          perc_same_classification=None,
                          activation_similarity=None):
    '''
    A function for printing results and logging them to W&B. 

    Parameters:
    train_or_eval (str): The string 'train' or 'eval' to indicate whether
                         the results are from training or evaluation.
    use_sae (Bool): True or False
    batch (int): index of batch whose results are logged
    dead_neurons_steps (int): after how many batches/epochs we measure dead neurons
    '''
    if train_or_eval == 'train':
        epoch_or_batch = 'batch'
        step = batch
    elif train_or_eval == 'eval':
        epoch_or_batch = 'epoch'
        step = epoch
    else:
        raise ValueError("train_or_eval needs to be 'train' or 'eval'.")

    # we don't print anything during training because there we deal with results batch-wise
    # printing for every batch would be too much, instead, if specified, these things will be logged to W&B
    if train_or_eval == 'eval':
        print(f"Model loss: {model_loss:.4f} | Model accuracy: {accuracy:.4f}")
        if use_sae:
            print(f"KLD: {kld:.4f} | Perc same classifications: {perc_same_classification:.4f}")
    if wandb_status:
        wandb.log({f"{train_or_eval}/model loss": model_loss, f"{train_or_eval}/model accuracy": accuracy, f"{epoch_or_batch}": step}, commit=False)
        if use_sae:
            wandb.log({f"{train_or_eval}/KLD": kld, 
                       f"{train_or_eval}/Perc same classifications": perc_same_classification, 
                       f"{epoch_or_batch}": step}, commit=False)
            # wandb doesn't accept tuples as keys, so we convert them to strings
            number_dead_neurons_wandb = {f"{train_or_eval}/Number_of_dead_neurons_on_{train_or_eval}_data/{k[0]}_{k[1]}": v for k, v in number_dead_neurons.items()}
            # merge two dictionaries and log them to W&B
            wandb.log({**number_dead_neurons_wandb, f"{epoch_or_batch}": step}, commit=False) # overview of number of dead neurons for all layers
        
    # We show per model layer evaluation metrics
    for name in sparsity_dict_1.keys(): 
    # the names are the same for the polysemanticity and relative sparsity dictionaries
    # hence, it suffices to iterate over the keys of the sparsity dictionary
        model_key = name[1] # can be "original" (model), "sae" (encoder output), "modified" (model)

        #if name[0] in layer_names or name[0] == 'fc2':
        if train_or_eval == 'eval':
            '''
            print("-------")
            if number_active_neurons is not None:
                print(f"Activated/total neurons: {number_active_neurons[name][0]:.2f} | {int(number_active_neurons[name][1])}")
            print(f"Dead neurons: {number_dead_neurons[name]}")
            print(f"Sparsity: {sparsity_dict[name]:.3f}")
            #if model_key == 'sae':
            print(f"Sparsity_1: {sparsity_dict_1[name]:.3f}")
            if mean_number_active_classes_per_neuron is not None:
                print(f"Mean and std of number of active classes per neuron: {mean_number_active_classes_per_neuron[name]:.4f} | {std_number_active_classes_per_neuron[name]:.4f}")
            if use_sae and activation_similarity is not None:
                print(f"Mean and std of feature similarity (L2 loss) between modified and original model: {activation_similarity[name[0]][0]:.4f} | {activation_similarity[name[0]][1]:.4f}")
            '''
            if use_sae and name[0] in sae_loss.keys() and model_key == 'sae':
                print(f"SAE loss, layer {name[0]}: {sae_loss[name[0]]:.4f} | SAE rec. loss, layer {name[0]}: {sae_rec_loss[name[0]]:.4f} | SAE l1 loss, layer {name[0]}: {sae_l1_loss[name[0]]:.4f}")
        if wandb_status:                    
            if train_or_eval == 'eval':
                wandb.log({f"{train_or_eval}/Sparsity/{model_key}_layer_{name[0]} sparsity": sparsity_dict[name], f"{epoch_or_batch}":step}, commit=False)
            if use_sae and name[0] in sae_loss.keys():
                wandb.log({f"{train_or_eval}/SAE_loss/layer_{name[0]} SAE loss": sae_loss[name[0]], f"{epoch_or_batch}":step}, commit=False)
                wandb.log({f"{train_or_eval}/SAE_loss/layer_{name[0]} SAE rec. loss": sae_rec_loss[name[0]], f"{epoch_or_batch}":step}, commit=False)
                wandb.log({f"{train_or_eval}/SAE_loss/layer_{name[0]} SAE l1 loss": sae_l1_loss[name[0]], f"{epoch_or_batch}":step}, commit=False)
                wandb.log({f"{train_or_eval}/SAE_loss/layer_{name[0]} SAE nrmse loss": sae_nrmse_loss[name[0]], f"{epoch_or_batch}":step}, commit=False)
                wandb.log({f"{train_or_eval}/SAE_loss/layer_{name[0]} SAE rmse loss": sae_rmse_loss[name[0]], f"{epoch_or_batch}":step}, commit=False)
             #if model_key == 'sae':
            wandb.log({f"{train_or_eval}/Sparsity/{model_key}_layer_{name[0]} sparsity_1": sparsity_dict_1[name], f"{epoch_or_batch}":step}, commit=False)  
            # Optional metrics
            if number_active_neurons is not None:
                #wandb.log({f"{train_or_eval}/Activation_of_neurons/{model_key}_layer_{name[0]} activated neurons": number_active_neurons[name][0], f"{epoch_or_batch}": step}, commit=False)            
                wandb.log({f"{train_or_eval}/Number_of_neurons/{model_key}_layer_{name[0]}": number_active_neurons[name][1], f"{epoch_or_batch}": step}, commit=False)
                #wandb.log({f"{train_or_eval}/Activation_of_neurons/{model_key}_layer_{name[0]} dead neurons": number_dead_neurons[name], f"{epoch_or_batch}": step}, commit=False)
            if use_sae and activation_similarity is not None:
                wandb.log({f"{train_or_eval}/Feature_similarity_L2loss_between_modified_and_original_model/{model_key}_layer_{name[0]} mean": activation_similarity[name[0]][0], f"{epoch_or_batch}":step}, commit=False) 
                wandb.log({f"{train_or_eval}/Feature_similarity_L2loss_between_modified_and_original_model/{model_key}_layer_{name[0]} std": activation_similarity[name[0]][1], f"{epoch_or_batch}":step}, commit=False) 
            if mean_number_active_classes_per_neuron is not None:
                wandb.log({f"{train_or_eval}/Active_classes_per_neuron/{model_key}_layer_{name[0]} mean": mean_number_active_classes_per_neuron[name], f"{epoch_or_batch}":step}, commit=False)
                wandb.log({f"{train_or_eval}/Active_classes_per_neuron/{model_key}_layer_{name[0]} std": std_number_active_classes_per_neuron[name], f"{epoch_or_batch}":step}, commit=False)
            
    # we only log results at the end of an epoch, which is when epoch is not None (eval mode) or when batch is the last batch (train mode)
    if wandb_status:
    #    if (epoch is not None) or (batch is not None and batch == num_batches):
    #        print("Logging results to W&B...")
        wandb.log({}, commit=True) # commit the above logs

    
def feature_similarity(activations,activation_similarity,device):
    '''
    calculates the feature similarity between the modified and original for one batch
    '''
    unique_layer_names = {key[0] for key in activations.keys()} # curly brackets denote a set --> only take unique values
    # alternatively: module_names = get_module_names(model)

    for name in unique_layer_names:
        # we get the activations of the specified layer
        original_activation = activations.get((name, 'original'), None).to(device)
        modified_activation = activations.get((name, 'modified'), None).to(device)

        sample_dist = torch.linalg.norm(original_activation - modified_activation, dim=1)
        dist_mean = sample_dist.mean().item()
        dist_std = sample_dist.std().item()   
        activation_similarity[name] = (dist_mean, dist_std)

        '''
        activations = [(k,v) for k, v in activations.items() if k[0] == name]
        # activations should be of the form (with len(activations)=3)
        # [((name, 'modified'), tensor), 
        #  ((name, 'original'), tensor)]
        # and if we inserted an SAE at the given layer with name, then 
        # ((name, 'sae'), tensors) is the first entry of activations
        # we check whether activations has the expected shape
        if (activations[-2][0][1] == "modified" and activations[-1][0][1] == "original"):
            activations_1 = activations[-1][1].to(device)
            activations_2 = activations[-2][1].to(device)
            print(activations_1.shape, activations_2.shape)
            sample_dist = torch.linalg.norm(activations_1 - activations_2, dim=1)
            dist_mean = sample_dist.mean().item()
            dist_std = sample_dist.std().item()   
            activation_similarity[name] = (dist_mean, dist_std)
        else:
            raise ValueError("Activations has the wrong shape for evaluating feature similarity.")
        '''
    return activation_similarity

def compute_sparsity(train_or_eval, 
                     sae_expansion_factor, 
                     number_active_neurons, 
                     active_classes_per_neuron=None, 
                     number_dead_neurons=None,
                     dead_neurons=None, 
                     num_classes=None):
    sparsity_dict = {}
    sparsity_dict_1 = {} # alternative sparsity
    number_active_classes_per_neuron = {}
    mean_number_active_classes_per_neuron = {}
    std_number_active_classes_per_neuron = {}

    for name in number_active_neurons.keys(): 
        model_key = name[1] # can be "original" (model), "sae" (encoder output), "modified" (model)
          
        average_activated_neurons = number_active_neurons[name][0]
        total_neurons = number_active_neurons[name][1] 
        # during training, we don't measure sparsity batch-wise in terms of dead neurons because the number 
        # of dead neurons monotonically decreases over batches --> thus the sparsity would change accordingly
        # making it hard to interpret
        if train_or_eval == 'eval':
            number_used_neurons = total_neurons - number_dead_neurons[name]
            # number of active neurons should be less than or equal to the number of used neurons
            if average_activated_neurons > number_used_neurons:
                raise ValueError(f"{name}: The number of active neurons ({average_activated_neurons}) is greater than the number of used neurons ({number_used_neurons}).")
            sparsity = 1 - (average_activated_neurons / number_used_neurons) 
            sparsity_dict[name] = sparsity
        #if model_key == 'sae':
        number_of_neurons_in_original_layer = total_neurons / sae_expansion_factor
        sparsity_1 = 1 - (average_activated_neurons / number_of_neurons_in_original_layer)
        sparsity_dict_1[name] = sparsity_1

        if active_classes_per_neuron is not None:
            # in the case, where dead neurons are measured over the same time frame as the active classes per neuron
            # (this is the case if we're in eval mode), we can check that active_classes_per_neuron is consistent with
            # the number of dead neurons. If this is the case during evaluation it like also is the case during training.
            if train_or_eval == 'eval':
                # First we check that the number of rows of the active_classes_per_neuron matrix
                # is equal to the number of neurons
                if active_classes_per_neuron[name].shape[0] != total_neurons:
                    raise ValueError(f"The number of rows of the active_classes_per_neuron matrix ({active_classes_per_neuron[name].shape[0]}) is not equal to the number of dead neurons ({total_neurons}).")

                # Now we remove the rows/neurons from the matrix which correspond to dead neurons
                a = active_classes_per_neuron[name][dead_neurons[name] == False]
                # we check whether the number of rows of the matrix with dead neurons removed is equal to the number of used neurons
                # this is equivalent to: the number of neurons which weren't active on any class is 
                # equal to the number of dead neurons
                if a.shape[0] != number_used_neurons:
                    raise ValueError(f"The number of rows of the active_classes_per_neuron matrix with dead neurons removed ({a.shape[0]}) is not equal to the number of used neurons ({number_used_neurons}).")

                # the number of dead neurons should correspond to 
                # the number of 0 rows in active_classes_per_neuron because a neuron is 
                # dead iff it is active on 0 classes throughout this epoch
                num_zero_rows = torch.sum(torch.all(active_classes_per_neuron[name] == 0, dim=1)).item()
                if number_dead_neurons[name] != num_zero_rows:
                    raise ValueError(f"{name}: The number of dead neurons ({number_dead_neurons[name]}) is not equal to the number of neurons ({num_zero_rows}) which are active on 0 classes.")
            
            # For each row i (each neuron), we count the number of distinct positive integers, i.e., the number 
            # of classes that have an activation above the threshold; .bool() turns every non-zero element into a 1, 
            # and every zero into a 0
            number_active_classes_per_neuron[name] = torch.sum(active_classes_per_neuron[name].bool(), dim=1)
            mean_number_active_classes_per_neuron[name] = number_active_classes_per_neuron[name].float().mean().item()
            if mean_number_active_classes_per_neuron[name] > num_classes:
                raise ValueError("The mean number of active classes per neuron ({}) is greater than the number of classes ({}).".format(mean_number_active_classes_per_neuron[name], num_classes))
            std_number_active_classes_per_neuron[name] = number_active_classes_per_neuron[name].float().std().item()
    
    return sparsity_dict, sparsity_dict_1, number_active_classes_per_neuron, mean_number_active_classes_per_neuron, std_number_active_classes_per_neuron

                                
def get_top_k_samples(top_k_samples, batch_top_k_values, batch_top_k_indices, eval_batch_idx, largest, k):
    '''
    top_k_samples = (previous_top_k_values of shape [k, #neurons], previous_top_k_indices of shape [k, #neurons], batch_size)
    batch_top_k_values = top k values of current batch, shape [k, #neurons]
    batch_top_k_indices = top k indices of current batch, shape [k, #neurons]
    --> Want new top k values and indices incorporating the current and previous batches
    '''
    # we add batch_idx*batch_size to every entry in the indices matrix to get the 
    # index of the corresponding image in the dataset. For example, if the index in the
    # batch is 2 and the batch index is 4 and batch size is 64, then the index of this 
    # image in the dataset is 4*64 + 2= 258
    # the batch size is the first dimension of 
    batch_top_k_indices += (eval_batch_idx - 1) * top_k_samples[2]
    # we merge the previous top k values and the current top k values --> shape: [2*k, #neurons]
    # then we find the top k values within this matrix
    top_k_values_merged = torch.cat((top_k_samples[0], batch_top_k_values), dim=0)
    # we also merge the indices
    top_k_indices_merged = torch.cat((top_k_samples[1], batch_top_k_indices), dim=0)
    # we find the top k values and indices of the merged top k values
    top_k_values_merged_new, top_k_indices_merged_new = torch.topk(top_k_values_merged, k=k, dim=0, largest=largest)
    # but top_k_indices_merged_new contains the indices of the top values within the top_k_values_merged
    # matrix, but we need to find the corresponding indices in top_k_indices_merged
    selected_indices = torch.gather(top_k_indices_merged, 0, top_k_indices_merged_new)
    # we store the newly found top values and indices
    return (top_k_values_merged_new, selected_indices, top_k_samples[2])

def rows_cols(i):
    '''
    Given i subplots to be plotted, infer a good number of rows and columns
    so that the figure doesn't look to high or wide
    '''
    # i >= 1, number of neurons
    # num_cols and num_rows are monotonically increasing (as i increases)
    # num_cols >= num_rows
    num_cols = math.floor(math.sqrt(i - 1)) + 1
    num_rows = math.ceil(i / num_cols)
    return num_cols, num_rows

def show_top_k_samples(val_dataloader, 
                       model_key, 
                       layer_name, 
                       top_k_samples, 
                       small_k_samples, 
                       n, 
                       folder_path=None, 
                       layer_names=None, 
                       params=None,
                       number_neurons=None, 
                       neuron_idx=None,
                       wandb_status=None,
                       dataset_name=None):
    '''
    n = sqrt(number of top samples to show for each neuron)
    Either specify: number_neurons = number of neurons to show samples for
    or specify: neuron_idx = index of neuron to show samples for
    '''
    if (number_neurons is not None) ^ (neuron_idx is None): # XOR
        raise ValueError("Either number_neurons or neuron_idx should be provided but not both.")

    if neuron_idx is not None:
        number_neurons = 1

    # matrix of shape [k=n**2, num_neurons] containing the indices of the top k samples for each neuron
    top_index_matrix = top_k_samples[(layer_name, model_key)][1]
    small_index_matrix = small_k_samples[(layer_name, model_key)][1]
    top_value_matrix = top_k_samples[(layer_name, model_key)][0]
    small_value_matrix = small_k_samples[(layer_name, model_key)][0]

    # we store the column indices of non-dead neurons --> we will not plot activating samples for dead neurons because this is pointless
    # we use the proxy that the top values are all zero for dead neurons
    non_zero_columns = torch.any(top_value_matrix != 0, dim=0) # --> tensor([True, False, True,...]), True wherever non-zero

    # the number of neurons we can plot is upper bounded by the number of non-dead neurons in the layer
    number_non_dead_neurons = torch.sum(non_zero_columns).item()
    number_neurons = min(number_neurons, number_non_dead_neurons)

    # based on the given number_neurons we find number of rows and columns
    # such that the number of rows and columns is as close as possible (approximately)
    # so that the figure doesn't become too wide or too high    
    num_cols, num_rows = rows_cols(number_neurons)

    fig = plt.figure(figsize=(18,9))#figsize=(num_rows*9, num_cols*18))
    outer_grid = fig.add_gridspec(num_rows, num_cols, wspace=0.1, hspace=0.3)
    fig.suptitle(f'{n**2} most and least activating samples for {number_neurons} neurons \n in {model_key, layer_name}, with activation values', fontsize=10)

    # find the column indices of the first number_neurons non-zero columns
    non_zero_columns_indices = torch.nonzero(non_zero_columns).squeeze()[:number_neurons]

    for i in range(num_rows):
        for j in range(num_cols):
            
            if i * num_cols + j >= number_neurons:
                break
            neuron_idx = non_zero_columns_indices[i * num_cols + j].item()
            
            # indices of the top k samples for the neuron with index neuron_idx
            top_indices = top_index_matrix[:,neuron_idx]
            small_indices = small_index_matrix[:,neuron_idx]
            top_values = top_value_matrix[:,neuron_idx]
            small_values = small_value_matrix[:,neuron_idx]

            outer_grid_1 = outer_grid[i,j].subgridspec(1, 3, wspace=0, hspace=0, width_ratios=[1, 0.1, 1])
            #or alternatively: outer_grid_1 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer_grid[i,j], wspace=0, hspace=0)

            # add ghost axis --> allows us to add a title
            ax_title_1 = fig.add_subplot(outer_grid_1[:])
            #ax_title.axis('off')
            ax_title_1.set(xticks=[], yticks=[])
            ax_title_1.set_title(f'Neuron {neuron_idx}\n', fontsize=9, pad=0) # we add a newline so that title is above the ones of the subplots
            #ax_title_1.set_box_aspect(0.5)	 

            ############################################################
            for a, indices, values, title in zip([0,2], [top_indices, small_indices], [top_values, small_values], ['Most', 'Least']):
                inner_grid = outer_grid_1[0, a].subgridspec(n, n, wspace=0, hspace=0)#, height_ratios=[1,1], width_ratios=[1,1])
                
                # add ghost axis --> allows us to add a title
                ax_title_2 = fig.add_subplot(inner_grid[:])
                #ax_title_2.axis('off')
                ax_title_2.set(xticks=[], yticks=[])
                ax_title_2.set_title(f'{title} activating', fontsize=8, pad=0.1)
                #ax_title_2.set_box_aspect(1)

                axs = inner_grid.subplots()  # Create all subplots for the inner grid.
                for (c, d), ax in np.ndenumerate(axs):
                    ax.set(xticks=[], yticks=[])
                    ax.set_box_aspect(1) # make the image square
                    idx = indices[c * n + d].item()
                    #print(len(val_dataloader.dataset))
                    #print(int(idx))
                    sample = val_dataloader.dataset[int(idx)]
                    if isinstance(sample, (list, tuple)) and len(sample) == 2:
                        img = sample[0]
                    elif isinstance(sample, dict) and len(sample) == 2 and list(sample.keys())[0] == "image" and list(sample.keys())[1] == "label":
                        img = sample["image"]
                    else: 
                        raise ValueError("Unexpected data format from dataloader")
                    
                    if dataset_name == 'mnist':
                        ax.imshow(img.permute(1, 2, 0), cmap='gray', aspect='auto')
                    elif dataset_name == 'tiny_imagenet':
                        img = img.astype(int)
                        ax.imshow(np.transpose(img, (1, 2, 0)))
                    else:
                        ax.imshow(img.permute(1, 2, 0), aspect='auto')
                    # turn off the activation value label temporarily
                    #ax.text(0,0, f'{values[c * n + d].item():.2f}', transform=ax.transAxes, fontsize=3,bbox=dict(facecolor='white', edgecolor='none',boxstyle='square,pad=0'))

    # show only the outside spines
    '''
    custom_spine_color = "white"
    for ax in fig.get_axes():
        ss = ax.get_subplotspec()
        ax.spines.top.set_visible(ss.is_first_row())
        ax.spines.bottom.set_visible(ss.is_last_row())
        ax.spines.left.set_visible(ss.is_first_col())
        ax.spines.right.set_visible(ss.is_last_col())

        ax.spines.top.set_color(custom_spine_color)
        ax.spines.bottom.set_color(custom_spine_color)
        ax.spines.left.set_color(custom_spine_color)
        ax.spines.right.set_color(custom_spine_color)
    '''
    #plt.show()
    #'''
    if wandb_status:
        wandb.log({f"most_and_least_activating_samples/{model_key}_{layer_name}":wandb.Image(plt)})
    else:
        folder_path = os.path.join(folder_path, 'visualize_top_samples')
        os.makedirs(folder_path, exist_ok=True)
        file_path = get_file_path(folder_path, layer_names, params, f'{model_key}_{layer_name}_{number_neurons}_top_{n**2}_samples.png')
        plt.savefig(file_path, bbox_inches='tight', pad_inches=0.1, dpi=300)
        print(f"Successfully stored top {n**2} samples for {number_neurons} neurons in {file_path}")
    plt.close()
    #'''

def process_sae_layers_list(sae_layers, original_model, training):
    '''
    Turns the string 'fc1_fc2__fc3_fc4' into the list ['fc1_fc2_fc3', 'fc1_fc2_fc3_fc4']
    --> For each element of the list (separated by underscores), the code will train an SAE on the last element of the list
        (if training = "True" and original_model = "False") and use pretrained SAEs on the preceding layers --> here:
        fc1_fc2_fc3 --> Train SAE on fc3 and use pretrained SAEs on fc1, and fc1_fc2 (i.e. on fc2 but given that SAE on fc1 is already trained)
        fc1_fc2_fc3_fc4 --> Train SAE on fc4 and use pretrained SAEs on fc1, fc1_fc2, fc1_fc2_fc3 
    '''
    pretrained_sae_layers_string, train_sae_layers_string = sae_layers.split("__")
    # Split the sub_string into a list based on '_'
    #pretrained_sae_layers_list = pretrained_sae_layers_string.split("_")
    train_sae_layers_list = train_sae_layers_string.split("_")

    if train_sae_layers_list != [""] and eval(original_model):
        raise ValueError(f"We use the original model but layers for training SAEs are provided: {train_sae_layers_list}")
    if train_sae_layers_list != [""] and not eval(original_model) and not eval(training):
        raise ValueError(f"We do inference through the SAE but layers for training SAEs are provided: {train_sae_layers_list}")
    if train_sae_layers_list == [""] and not eval(original_model) and eval(training):
        raise ValueError(f"We train SAEs but no layers for training SAEs are provided.")

    if train_sae_layers_list != [""]:
        sae_layers_list = []
        for i in range(len(train_sae_layers_list)):
            if pretrained_sae_layers_string == "":
                name = train_sae_layers_list[i]
            else:
                name = pretrained_sae_layers_string + "_" + train_sae_layers_list[i]
            sae_layers_list.append(name)
            pretrained_sae_layers_string = name
    else:
        sae_layers_list = [sae_layers]  
    return sae_layers_list

def activation_histograms(activations, folder_path, layer_names, params, wandb_status, targets=None):
    '''
    activations is a dictionary of the form {(layer_name, model_key): tensor}
    where tensor has shape [number of samples in one epoch, number of neurons in layer]
    For example, for MNIST we have 10,000 samples in the eval dataset
    '''
    for key, tensor in activations.items():
        number_neurons = tensor.shape[1]
        # Visualize the histograms of at most 32 neurons
        number_neurons_to_visualize = min(32, number_neurons)
        cols, rows = rows_cols(number_neurons_to_visualize)
        fig = plt.figure(figsize=(18,12))
        plt.suptitle(f"Histograms of neuron activations, {key}")
        for i in range(number_neurons_to_visualize):
            neuron_activations = tensor[:, i]
            if targets is not None:
                targets_for_loop = targets.clone()
            plt.subplot(rows, cols, i+1)
            perc_pos_activations = (neuron_activations > 0).sum() / len(neuron_activations) * 100
            perc_zero_activations = (neuron_activations == 0).sum() / len(neuron_activations) * 100
            perc_neg_activations = (neuron_activations < 0).sum() / len(neuron_activations) * 100
            if (neuron_activations < 0).any():
                plt.axvline(x=0, color='red')
            else:
                # if we don't have negative activation value we are likely looking at the output of ReLU
                # hence, there might be a lot of zeros --> we exclude those, so that the histogram does not
                # have one super high bar at zero making the rest of the histogram hard to see
                # instead we print the percentage of positive activations
                if targets is not None:                
                    targets_for_loop = targets_for_loop[neuron_activations > 0]
                neuron_activations = neuron_activations[neuron_activations > 0]
            if targets is not None:
                unique_targets = np.unique(targets_for_loop)
                # Plot histogram with stacked bars for each target value
                plt.hist([neuron_activations[targets_for_loop == target] for target in unique_targets], bins=100, label=[f'{target}' for target in unique_targets], stacked=True)
                #plt.legend()
                name = "activation_histogram_with_classes"
                handles, labels = plt.gca().get_legend_handles_labels()
            else:
                plt.hist(neuron_activations, bins=100, color='dodgerblue')
                name = "activation_histogram"
            plt.text(0.95, 0.95, f'>0: {perc_pos_activations:.0f}%\n =0: {perc_zero_activations:.0f}% \n <0: {perc_neg_activations:.0f}%', horizontalalignment='right', verticalalignment='top', transform=plt.gca().transAxes)
            # the transform statement specifies that the position of the text item is relative to the plot's axes not in 
            # absolute x and y values
            plt.xlabel('Activation value')
            plt.ylabel('No. of samples')
            plt.title(f'Neuron {i}')
        fig.tight_layout(pad=1.0)
        if targets is not None:
            fig.legend(handles, labels, loc='upper right', ncol=5)
        if wandb_status:
            wandb.log({f"eval/{name}/{key[0]}_{key[1]}":wandb.Image(plt)})
        # store the figure also if we use the cluster because resolution with W&B might not be high enough
        os.makedirs(folder_path, exist_ok=True)
        file_path = get_file_path(folder_path=folder_path, layer_names=layer_names, params=params, file_name=f'{name}_{key[0]}_{key[1]}.png')
        plt.savefig(file_path, dpi=300)
        plt.close()
        print(f"Successfully stored {name} of layer {key[0]}, model {key[1]} in {file_path}")

def processed_optim_image(image_data, version, mean, std, first_image):
    if version == '1': # standardization and clamp
        # standardize optim_image
        image_data = (image_data - image_data.mean()) / image_data.std()
        # make optim_image have the same mean and std as the MNIST training set
        image_data = image_data * std + mean
        image_data = torch.clamp(image_data, 0, 1)
    elif version == '2': # standardization but without clamp
        image_data = (image_data - image_data.mean()) / image_data.std()
        image_data = image_data * std + mean
    else:
        raise ValueError("Version should be 1 or 2.")
    return image_data

def plot_lucent_explanations(model, layer_names, params, folder_path, wandb_status, num_units):
    folder_path = os.path.join(folder_path, 'lucent_explanations')
    os.makedirs(folder_path, exist_ok=True)
    layer_list = []
    if layer_names == "None__": # if we use the original model
        print("Not computing maximally activating explanations for the original model. If you want to do so, please specify layers manually in the plot_lucent_explanations function.")
    else: # if we use SAE
        # count the number of non-empty strings in layer_names.split("_")
        if sum([1 for name in layer_names.split("_") if name != ""]) > 1:
            print("Should we compute maximally activating explanations for all SAE layers? As of right now, nothing is computed")
        else:
            for name in layer_names.split("_"):
                if name != "":
                    # for example name = 'layer1.0.conv1'
                    # replace '.' with '_' to get 'layer1_0_conv1'
                    name = name.replace('.', '_')
                    layer_list += [f"{name}_0", f"{name}_1_encoder", f"{name}_1_decoder"]
                    # F.e. 'layer1_0_conv1_0' is output of original model, 'layer1_0_conv1_1_encoder is output of SAE encoder
                    # and 'layer1_0_conv1_1_decoder' is output of SAE decoder

    for layer_name in layer_list:
        cols, rows = rows_cols(num_units)
        fig = plt.figure(figsize=(18,12))
        plt.suptitle(f"Maximally activating explanations, {layer_name}")
        for i in range(num_units):
            plt.subplot(rows, cols, i+1)
            img = render.render_vis(model, "{}:{}".format(layer_name, i), show_image=False)[0] 
            # the output is a list of images for different thresholds when to stop optimizing --> we only use one threshold 
            # --> only one image in the list, which we get through [0]
            img = img.squeeze(0)  # remove batch dimension (why is there a batch dimension in the first place?)
            plt.imshow(img)
            plt.axis('off')
            plt.title(f'Neuron {i}')
        fig.tight_layout(pad=1.0)
        if wandb_status:
            wandb.log({f"eval/maximally_activating_explanations/{layer_name}":wandb.Image(plt)})
        # store the figure also if we use the cluster because resolution with W&B might not be high enough
        file_path = get_file_path(folder_path=folder_path, layer_names=layer_names, params=params, file_name=f'lucent_explanations_{layer_name}_for_{num_units}_neurons.png')
        plt.savefig(file_path, dpi=300)
        plt.close()
        print(f"Successfully stored maximally activating explanations of layer {layer_name} in {file_path}")

def update_histogram(histogram_info, name, model_key, output, device, output_2=None):
    # we unpack the values contained in the dictionary
    histogram_matrix = histogram_info[(name,model_key)][0]
    histogram_matrix = histogram_matrix.to(device)
    top_values = histogram_info[(name,model_key)][1]
    small_values = histogram_info[(name,model_key)][2]
    num_bins = histogram_matrix.shape[0]
    num_units = histogram_matrix.shape[1]

    # we get the activations of the current batch 
    if output_2 is not None and model_key=='sae': # we get the pre-relu encoder output
        activations = output_2
    elif output_2 is None and model_key!='sae':
        activations = output
    else:
        raise ValueError(f"model_key is {model_key} but output_2 is {output_2}")

    # we only consider the first n units
    activations = activations[:,:num_units]
    activations = activations.to(device)
    
    # we compute the histogram of the activations
    for unit in range(activations.size(1)):
        top_value = top_values[unit].item() # these are invariant over the batches and hence we will hae the same bins over all batches
        small_value = small_values[unit].item() 
        histogram_matrix[:, unit] += torch.histc(activations[:,unit], bins=num_bins, min=small_value, max=top_value)

    # we write the updated histogram_matrix back into the dictionary
    histogram_info[(name,model_key)] = (histogram_matrix, top_values, small_values)
    return histogram_info


def activation_histograms_2(histogram_info, folder_path, layer_names, params, wandb_status, num_units):
    name = "activation_histograms"
    folder_path = os.path.join(folder_path, name)
    os.makedirs(folder_path, exist_ok=True)
    for key, v in histogram_info.items():
        histogram_matrix = v[0]
        top_values = v[1]
        small_values = v[2]
        num_bins = histogram_matrix.shape[0] 
        num_units = histogram_matrix.shape[1]
        cols, rows = rows_cols(num_units)
        fig = plt.figure(figsize=(18,12))
        plt.suptitle(f"Histograms of neuron activations, {key}")
        for i in range(num_units):
            edges = torch.linspace(small_values[i], top_values[i], num_bins + 1)
            plt.subplot(rows, cols, i+1)
            plt.stairs(values=histogram_matrix[:, i].cpu().numpy(), edges=edges, fill=True)
            plt.xlabel('Activation value')
            plt.ylabel('No. of samples')
            plt.title(f'Neuron {i}')
        fig.tight_layout(pad=1.0)
        if wandb_status:
            wandb.log({f"eval/{name}/{key[0]}_{key[1]}":wandb.Image(plt)})
        # store the figure also if we use the cluster because resolution with W&B might not be high enough
        file_path = get_file_path(folder_path=folder_path, layer_names=layer_names, params=params, file_name=f'{name}_{key[0]}_{key[1]}.png')
        plt.savefig(file_path, dpi=300)
        plt.close()
        print(f"Successfully stored {name} of layer {key[0]}, model {key[1]} in {file_path}")