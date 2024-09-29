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
import pandas as pd
import webdataset as wds
from lucent.modelzoo import inceptionv1
import glob
import re
import logging
import socket
from datetime import datetime, timedelta
from torch.autograd.profiler import record_function
import gc
import time
from filelock import FileLock
from filelock import Timeout
from einops import rearrange
import tarfile
import shutil
from nnsight import NNsight
import pickle

from dataloaders.tiny_imagenet import *
from dataloaders.tiny_imagenet import _add_channels
from models.custom_mlp import *
from models.custom_cnn import *
from models.sae_conv import SaeConv
from models.sae_mlp import SaeMLP
from models.gated_sae import GatedSae
from losses.sparse_loss import SparseLoss, GatedSAELoss
from supplementary.dataset_stats import print_dataset_stats
#from dataloaders.imagenet import *
from dataloaders.old_imagenet_classnames import imagenet_classnames
from machine_interpretability.stimuli_generation import mis_utils, sg_utils

class ConstrainedAdam(torch.optim.Adam):
    """
    A variant of Adam where some of the parameters are constrained to have unit norm.
    # from: https://github.com/saprmarks/dictionary_learning/blob/main/training.py     
    """
    def __init__(self, params, constrained_params, lr):
        super().__init__(params, lr=lr, betas=(0.9, 0.999))
        # CAREFUL: The authors of this function set constrained_params = model.decoder.parameters()
        # In their code, these parameters only contain the weights. But in my equivalent 
        # implementation, I also have a decoder bias. So I use this function with
        # constrained_params = model.decoder.weights
        # Otherwise when trying to normalize the decoder bias I would get an error
        # because bias = 0 initially so dividing by its norm would result in NaN
        self.p = constrained_params
    
    def step(self, closure=None):
        with torch.no_grad():
            #for p in constrained_params:
            if self.p.grad is not None: # f.e. if we remove the rec loss for debugging, then we only have loss on SAE encoder output. In that
                    # case, when applying loss.backward(), the decoder weights will not have a gradient, because they obviously don't matter in 
                    # minimizing the L1 loss (and because they are not "backward" of the loss), so we would get an error here...
                    # Also, if we don't adjust the decoder weights, they still have unit norm as by the initialization
                if self.p.norm(dim=0).min() < 1e-6:
                    logging.warning(f"Constrained parameter {self.p} has a norm smaller than 1e-6")
                normed_p = self.p / self.p.norm(dim=0, keepdim=True)
                # project away the parallel component of the gradient
                self.p.grad -= (self.p.grad * normed_p).sum(dim=0, keepdim=True) * normed_p
        super().step(closure=closure) # step of Adam
        with torch.no_grad():
            #for p in constrained_params:
            # renormalize the constrained parameters
            self.p /= self.p.norm(dim=0, keepdim=True)
        #print(p.grad) --> zero tensor

def get_optimizer(optimizer_name, model, learning_rate):
    if optimizer_name == 'adam':
        return torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9,0.9999)), None
    elif optimizer_name == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=learning_rate), None
    elif optimizer_name == 'sgd_w_scheduler':
        # here we also use momentum (for ResNet-18)
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)    
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        return optimizer, scheduler
    elif optimizer_name == 'constrained_adam':
        return ConstrainedAdam(model.parameters(), model.decoder.weight, lr=learning_rate), None
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

class CustomCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CustomCrossEntropyLoss, self).__init__()

    def forward(self, logits, targets):
        """
        Args:
            logits (torch.Tensor): Predicted probabilities (logits) of shape [batch_size, num_classes].
            targets (torch.Tensor): Ground truth class indices of shape [batch_size].
        """
        # Ensure the logits are normalized (each row should sum to 1)
        # doesnt work in trace context of nnsight
        #if not torch.allclose(logits.sum(dim=1), torch.ones(logits.size(0))):
        #    raise ValueError("Logits should be normalized such that each row sums to 1")

        # Step 1: Extract the logits corresponding to the correct class for each sample
        #correct_class_logits = logits[range(logits.size(0)), targets]
        correct_class_logits = torch.gather(logits, 1, targets.unsqueeze(1)).squeeze(1)

        # Step 2: Compute the negative log-likelihood of these logits
        # add a small value to avoid that we take log of 0
        negative_log_likelihoods = -torch.log(correct_class_logits + 1e-40)

        # Step 3: Calculate the mean of these negative log-likelihoods for the final loss
        loss = torch.mean(negative_log_likelihoods)

        return loss

def get_criterion(criterion_name):
    if criterion_name == 'cross_entropy':
        return nn.CrossEntropyLoss()
    elif criterion_name == 'sae_loss':
        return SparseLoss()
    elif criterion_name == 'gated_sae_loss':
        return GatedSAELoss()
    elif criterion_name == 'negative_log_likelihood':
        return CustomCrossEntropyLoss()
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
    
def get_file_path(folder_path=None, sae_layer=None, params=None, file_name=None, params2=None):
    '''
    params and params2 expect a dictionary of parameters
    '''
    if file_name is not None:
        if file_name.startswith('.'): 
            ending = file_name 
        else:
            ending = f'_{file_name}'
    else:
        ending = f'_{file_name}'

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
            file_name = f'{sae_layer}_{params}_{params2}{ending}'
        else:
            file_name = f'{sae_layer}_{params}{ending}'
    else:
        file_name = f'{sae_layer}{ending}'

    if folder_path is None:
        file_path = file_name
    else:
        file_path = os.path.join(folder_path, file_name)
    return file_path
    
def save_model_weights(model, 
                       folder_path, 
                       sae_layer=None, # layer_name is used for SAE models, because SAE is trained on activations of a specific layer
                       params=None):
    os.makedirs(folder_path, exist_ok=True) # create folder if it doesn't exist
    file_path = get_file_path(folder_path, sae_layer, params, 'model_weights.pth')
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
    if model_name != "resnet18" and model_name != "resnet50" and model_name != "inceptionv1":
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
    
    elif model_name == 'inceptionv1':
        # https://github.com/greentfrapp/lucent/blob/dev/lucent/modelzoo/inceptionv1/InceptionV1.py
        # https://github.com/greentfrapp/lucent/tree/dev 
        return torchvision.models.googlenet(pretrained=True, aux_logits=True) #, transform_input=True) # gradients and thus IE values are slightly bigger
        #return inceptionv1(pretrained=True)
    
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
    elif model_name == 'gated_sae':
        return GatedSae(img_size, expansion_factor)
    else:
        raise ValueError(f"Unsupported model: {model_name}")


#def normalize(x):
#    return x * 255 - 117

def imagenet_transform():
    normalize = lambda x: x * 255 - 117
    img_size = 229 
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return transform

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
        ''' # Roland's original code suggestion
        dataset = (
            wds.WebDataset(datadir)
            .shuffle(True)
            .decode("pil")
            .to_tuple("jpeg.jpg;png.png jpeg.cls __key__")
            .map_tuple(transform, lambda x: x, lambda x: x)
            .batched(batch_size, partial=False)
        )
        dataloader = wds.WebLoader(
            dataset,
            batch_size=None,
            shuffle=False,
            num_workers=8,
        )
        '''

        '''
        SPLITTING THE TRAIN DATA INTO TRAIN AND VAL
        # Use glob to find files matching the pattern
        train_files = glob.glob(os.path.join(datadir, "imagenet-train-*"))
        #val_files = glob.glob(os.path.join(datadir, "imagenet-val-*"))

        # For training we use the first 141 folders: 000000 to 000140
        # For evaluation we use the folders 000141 to 000146
        
        # Define a regular expression patterns to match "imagenet-train-000x" where x is from 000 to 140
        pattern1 = r"imagenet-train-0000([0-9]{1,2})" # Matches numbers from 000000 to 000099, equivalent to 0000[0-9][0-9]
        pattern2 = r"imagenet-train-0001[0-3][0-9]" # Matches numbers from 000100 to 000139
        pattern3 = r"imagenet-train-000140"
        train_pattern = re.compile(rf"{pattern1}|{pattern2}|{pattern3}") # Combine the regex patterns 
        
        # We do the same for x from 141 to 146
        val_pattern = re.compile(r"imagenet-train-00014[1-6]") # Matches numbers from 000141 to 000146

        # Filter the list of folder names based on the regex pattern
        train_train_files = [folder for folder in train_files if train_pattern.search(folder)]
        train_val_files = [folder for folder in train_files if val_pattern.search(folder)]
        '''

        '''
        # On shuffling a web dataset

        .shuffle(n): described here: https://webdataset.github.io/webdataset/gettingstarted/ 

        Two further helpful resources:
        https://github.com/webdataset/webdataset/issues/62 
        https://github.com/webdataset/webdataset/issues/71 
        '''

        if directory_path.startswith('/lustre'):
            num_workers = 8
            condor_scratch_dir = os.getenv('_CONDOR_SCRATCH_DIR')
            if os.path.join(condor_scratch_dir, "ImageNet2012-webdataset") in os.listdir(condor_scratch_dir):
                datadir = os.path.join(condor_scratch_dir, "ImageNet2012-webdataset")
                # print(f"_CONDOR_SCRATCH_DIR is set to: {condor_scratch_dir}")
                #print("Condor scratch directory contents:", os.listdir(condor_scratch_dir))
                #print("Subdirectories of ImageNet2012-webdataset:", os.listdir(os.path.join(condor_scratch_dir, "ImageNet2012-webdataset")))
            else:
                datadir = "/fast/rzimmermann/ImageNet2012-webdataset"
  
            train_files = glob.glob(os.path.join(datadir, "imagenet-train-*"))
            # reduce size for now
            #pattern = re.compile(r"imagenet-train-0000[0-2][0-9]")
            #pattern = re.compile(r"imagenet-train-000000") 
            #pattern = re.compile(r"imagenet-train-000010") #[0-5]") # used for training SAEs on layer mixed3a
            # Filter the list of folder names based on the regex pattern
            #train_files = [folder for folder in train_files if pattern.search(folder)]

            val_files = glob.glob(os.path.join(datadir, "imagenet-val-*"))


            # Get a list of all files in the directory
            #all_files = os.listdir(datadir)
            # Filter to get only the files starting with "imagenet-train-"
            #train_files = [os.path.join(datadir, f) for f in all_files if f.startswith("imagenet-train-")]
            # Sort the list to maintain consistent order
            #train_files.sort()

            train_dataset = (
                wds.WebDataset(train_files, shardshuffle=train_shuffle)#True) # shuffle different shards (tar folders)
                .shuffle(2000) # just trying this number here let's see if it works, check if overfitting or cyclic up and down of train loss...
                .decode("pil") 
                .to_tuple("jpeg.jpg;png.png jpeg.cls __key__")
                .map_tuple(imagenet_transform(), lambda x: x, lambda x: x)
                .batched(batch_size, partial=False) # partial=False, does not include the last batch if it contains fewer elements than batch_size
            )
            val_dataset = (
                wds.WebDataset(val_files, shardshuffle=eval_shuffle) # don't shuffle different shards/tar folder
                #.shuffle(2000) 
                .decode("pil")
                .to_tuple("jpeg.jpg;png.png jpeg.cls __key__")
                .map_tuple(imagenet_transform(), lambda x: x, lambda x: x)
                .batched(batch_size, partial=False)
            )
            train_dataloader = wds.WebLoader(
                train_dataset,
                batch_size=None,
                shuffle=False, # no batching and shuffling done in dataloader but in dataset directly
                num_workers=num_workers,
            )
            val_dataloader = wds.WebLoader(
                val_dataset,
                batch_size=None,
                shuffle=False,
                num_workers=num_workers,
            )


        elif directory_path.startswith('C:'):
            # for testing purposes we use a small local part of Imagenet
            # we do not use a webdataset here, because this lead to some issues
            datadir = 'C:\\Users\\Jasper\\Downloads\\Master thesis\\Imagenet one folder\\imagenet-unpacked-train-000146'
            image_dataset = datasets.ImageFolder(root=datadir, transform=imagenet_transform())
            train_dataloader = DataLoader(image_dataset, batch_size=batch_size, shuffle=False)
            val_dataloader = train_dataloader
        else:
            raise ValueError(f"Unexpected directory path {directory_path}") 
        
        print("Data directory ", datadir)
         
        category_names = imagenet_classnames

        '''         
        # Define transformations to be applied to the images
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        '''
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

    return train_dataloader, train_dataloader, category_names, img_size # train_dataloader, val_dataloader

    
def store_feature_maps(activations, folder_path, params=None):
    os.makedirs(folder_path, exist_ok=True) # create the folder

    # store the intermediate feature maps
    for name in activations.keys():
        activation = activations[name]

        # if activation is empty, we give error message
        if activation.nelement() == 0:
            raise ValueError(f"Activation of layer {name} is empty")
        else:
            file_path = get_file_path(folder_path, sae_layer=[name], params=params, file_name='activations.h5')
            # Store activations to an HDF5 file
            with h5py.File(file_path, 'w') as h5_file:
                h5_file.create_dataset('data', data=activation.cpu().numpy())

def store_batch_feature_maps(activation, num_samples, name, folder_path, params=None):
    os.makedirs(folder_path, exist_ok=True) # create the folder

    # if activation is empty, we give error message
    if activation.nelement() == 0:
        raise ValueError(f"Activation of layer {name} is empty")
    else:
        file_path = get_file_path(folder_path, sae_layer=[name], params=params, file_name='activations.h5')
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
    sae_layer = [name for name, _ in model.named_modules()]
    sae_layer = list(filter(None, sae_layer)) # remove emtpy strings
    return sae_layer

def get_classifications(output, category_names, imagenet):
    # if output is already a prob distribution (f.e. if last layer of network is softmax)
    # then we don't apply softmax. Applying softmax twice would be wrong.
    if torch.allclose(output.sum(dim=1), torch.tensor(1.0), atol=1e-3) and (output.min().item() >= 0) and (output.max().item() <= 1):
        prob = output
    else:
        prob = F.softmax(output, dim=1)
    scores, class_ids = prob.max(dim=1)

    if imagenet:
        class_ids -= 1 # imagenet class ids start at 1, we want them to start at 0 to look up the names in a list (whose index starts at 0)

    if category_names is not None:
        category_list = [category_names[index] for index in class_ids]
    else:
        category_list = None

    return scores, category_list, class_ids

def show_classification_with_images(dataloader,
                                    class_names, 
                                    wandb_status,
                                    directory_path,
                                    folder_path=None,
                                    sae_layer=None,
                                    model=None,
                                    device=None,
                                    output=None,
                                    output_2=None,
                                    params=None):
    '''
    This function either works with available model output or the model can be used to generate the output.
    '''
    os.makedirs(folder_path, exist_ok=True)
    file_path = get_file_path(folder_path, sae_layer, params, 'classif_visual_original.png')
    
    n = 10  # show only the first n images, 
    # for showing all images in the batch use len(predicted_classes)

    for batch in dataloader:
        input_images, target_ids, _ = process_batch(batch, directory_path=directory_path)
        break # we only need the first batch

    input_images, target_ids = input_images[:n], target_ids[:n] # only show the first 10 images
    input_images, target_ids = input_images.to(device), target_ids.to(device)

    if model is not None:
        output = model(input_images)
        output = F.softmax(output, dim=1)

    if "tiny_imagenet" not in folder_path and "imagenet" in folder_path: # if we only do: "imagenet" in folder_path, this also includes "tiny_imagenet"
        imagenet = True
        target_ids -= 1 # imagenet class ids start at 1, we want them to start at 0 to look up the names in a list (whose index starts at 0)
    else:
        imagenet = False
    
    scores, predicted_classes, _ = get_classifications(output, class_names, imagenet)

    if output_2 is not None:
        scores_2, predicted_classes_2, _ = get_classifications(output_2, class_names, imagenet)
        file_path = get_file_path(folder_path, sae_layer, params, 'classif_visual_original_modified.png')
    
    fig, axes = plt.subplots(1, n + 1, figsize=(20, 3))

    # Add a title column to the left
    title_column = 'True\nOriginal Prediction\nModified Prediction'
    axes[0].text(0.5, 0.5, title_column, va='center', ha='center', fontsize=8, wrap=True)
    axes[0].axis('off')

    for i in range(n):
        if "tiny_imagenet" in folder_path:
            # input_images[i].shape) --> torch.Size([3, 64, 64])
            img = input_images[i].cpu().numpy()
            #img = _add_channels(img) 
            # we don't need to unnormalize the image
            img = img.astype(int)
            axes[i + 1].imshow(np.transpose(img, (1, 2, 0)))
        elif "mnist" in folder_path:
            #mean=(0.1307,)
            #std=(0.3081,)
            #img = input_images[i] * np.array(std) + np.array(mean)
            #img = np.clip(img, 0, 1)
            img = input_images[i].cpu().numpy()
            axes[i+1].imshow(img.squeeze(), cmap='gray')
        elif "tiny_imagenet" not in folder_path and "imagenet" in folder_path:
            img = input_images[i].cpu().numpy()
            img = (img + 117) / 255 # unnormalize image
            axes[i+1].imshow(np.transpose(img, (1, 2, 0)))
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
                                   sae_layer, 
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
        if name[0] == sae_layer.split("_")[-1] or name[0] == 'fc3':
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
        file_path = get_file_path(folder_path, sae_layer, params, 'active_classes_per_neuron.png')
        plt.savefig(file_path)
        plt.close()
        print(f"Successfully stored active classes per neuron plot in {file_path}")

def plot_neuron_activation_density(active_classes_per_neuron, 
                                   sae_layer, 
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
        if name[0] == sae_layer.split("_")[-1] or name[0] == 'fc3':
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
        file_path = get_file_path(folder_path, sae_layer, params, 'neuron_activation_density.png')
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
                       sae_layer=None, 
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
                            sae_layer, 
                            params, 
                            epochs,
                            lambda_sparse, 
                            expansion_factor, 
                            batch_size,
                            optimizer_name, 
                            learning_rate,
                            rec_loss=None, 
                            scaled_l1_loss=None, 
                            nrmse_loss=None,
                            rmse_loss=None,
                            aux_loss=None,
                            sparsity=None,
                            var_expl=None,
                            perc_dead_units=None,
                            loss_diff=None,
                            median_mis=None):
                            #sparsity_1=None):
    # params doesn't contain lambda_sparse, expansion factor, learning rate etc. because we 
    # want a uniform file name for all different values
    file_path = get_file_path(folder_path=folder_path,
                            sae_layer=sae_layer,
                            params=params,
                            file_name='sae_eval_results.csv')
    file_exists = os.path.exists(file_path)
    columns = ["lambda_sparse", 
               "expansion_factor", 
               "batch_size",
               "optimizer_name",
               "learning_rate",
               "rec_loss", 
               "l1_loss", 
               "nrmse_loss", 
               "rmse_loss", 
               "aux_loss",
               "rel_sparsity",
               "var_expl",
               "perc_dead_units",
               "loss_diff",
               "median_mis",
               "epochs"]#, "rel_sparsity_1"]
    
    if file_exists:
        # We use file locking to ensure that only one process at a time
        # writes to the file. For example, when training many models at once,
        # several of them might want to write to the file at once, leading to an error.
        lock_path = file_path + '.lock'  # Create a lock file
        lock = FileLock(lock_path) # Create a lock object
        lock_timeout = 10 # Specify a timeout (in seconds)
    
        try:
            with lock.acquire(timeout=lock_timeout):
                # Read the file
                try:
                    df = pd.read_csv(file_path)
                except Exception as e:
                    print(f"An error occurred while accessing the file: {e}")
                
                column_names = df.columns.tolist()
                if column_names != columns:
                    # this might happen if we added some quantities/columns which previously weren't computed
                    create_new_file = True
                else:
                    create_new_file = False

        except Timeout:
            print(f"Failed to acquire lock after {lock_timeout} seconds")

    '''    
    if file_exists:
        df = pd.read_csv(file_path)
        column_names = df.columns.tolist()
        if column_names != columns:
            # this might happen if we added some quantities/columns which previously weren't computed
            create_new_file = True 
        else:
            create_new_file = False
    '''

    if not file_exists or create_new_file:
        with open(file_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=columns)
            writer.writeheader()
            writer.writerow({columns[0]: lambda_sparse,
                            columns[1]: expansion_factor,
                            columns[2]: batch_size,
                            columns[3]: optimizer_name,
                            columns[4]: learning_rate,
                            columns[5]: rec_loss,
                            columns[6]: scaled_l1_loss,
                            columns[7]: nrmse_loss,
                            columns[8]: rmse_loss,
                            columns[9]: aux_loss,
                            columns[10]: sparsity,
                            columns[11]: var_expl,
                            columns[12]: perc_dead_units,
                            columns[13]: loss_diff,
                            columns[14]: median_mis,
                            columns[15]: epochs})
    else:
        # Read the existing CSV file
        with open(file_path, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile, fieldnames=columns)
            rows = list(reader)

            # Check if the combination of lambda_sparse, expansion_factor, batch_size, optimizer_name, learning_rate already exists
            combination_exists = any(row["lambda_sparse"] == str(lambda_sparse) 
                                     and row["expansion_factor"] == str(expansion_factor)
                                      and row["batch_size"] == str(batch_size)
                                       and row["optimizer_name"] == str(optimizer_name)
                                        and row["learning_rate"] == str(learning_rate) 
                                         and row["epochs"] == str(epochs)
                                          for row in rows)

            # If the combination exists, update rec_loss, l1_loss, relative sparsity
            if combination_exists:
                for row in rows:
                    if row["lambda_sparse"] == str(lambda_sparse) and row["expansion_factor"] == str(expansion_factor) and row["batch_size"] == str(batch_size) and row["optimizer_name"] == str(optimizer_name) and row["learning_rate"] == str(learning_rate) and row["epochs"] == str(epochs):
                        if rec_loss is not None:
                            row["rec_loss"] = str(rec_loss)
                        if scaled_l1_loss is not None:
                            row["l1_loss"] = str(scaled_l1_loss)
                        if nrmse_loss is not None:
                            row["nrmse_loss"] = str(nrmse_loss)
                        if rmse_loss is not None:
                            row["rmse_loss"] = str(rmse_loss)
                        if aux_loss is not None:
                            row["aux_loss"] = str(aux_loss)
                        if sparsity is not None:
                            row["rel_sparsity"] = str(sparsity)
                        if var_expl is not None:
                            row["var_expl"] = str(var_expl)
                        if perc_dead_units is not None:
                            row["perc_dead_units"] = str(perc_dead_units)
                        if loss_diff is not None:
                            row["loss_diff"] = str(loss_diff)
                        if median_mis is not None:
                            row["median_mis"] = str(median_mis)
                        #if sparsity_1 is not None:
                        #    row["rel_sparsity_1"] = str(sparsity_1)
                        break
            else:
                # If the combination doesn't exist, add a new row
                rows.append({"lambda_sparse": str(lambda_sparse), 
                            "expansion_factor": str(expansion_factor), 
                            "batch_size": str(batch_size),
                            "optimizer_name": str(optimizer_name),
                            "learning_rate": str(learning_rate),
                            "rec_loss": str(rec_loss), 
                            "l1_loss": str(scaled_l1_loss),
                            "nrmse_loss": str(nrmse_loss),
                            "rmse_loss": str(rmse_loss),
                            "aux_loss": str(aux_loss),
                            "rel_sparsity": str(sparsity),
                            "var_expl": str(var_expl),
                            "perc_dead_units": str(perc_dead_units),
                            "loss_diff": str(loss_diff),
                            "median_mis": str(median_mis),
                            "epochs": str(epochs)})
                            #"rel_sparsity_1": str(sparsity_1)})

        # Write the updated data back to the CSV file
        with open(file_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=columns)
            writer.writerows(rows)
    print(f"Successfully stored SAE eval results in {file_path}")

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
    perc_dead_neurons = {}
    for key, tensor in dead_neurons.items():
        perc_dead_neurons[key] = tensor.sum().item() / len(tensor)
    return perc_dead_neurons

def print_and_log_results(epoch_mode,
                          model_loss,
                          loss_diff,
                          accuracy,
                          use_sae,
                          wandb_status,
                          sparsity_dict=None,
                          mean_number_active_classes_per_neuron=None,
                          std_number_active_classes_per_neuron=None,
                          number_active_neurons=None,
                          sparsity_dict_1=None,
                          perc_dead_neurons=None,
                          batch=None,
                          epoch=None,
                          sae_loss=None,
                          sae_rec_loss=None,
                          sae_l1_loss=None,
                          sae_nrmse_loss=None,
                          sae_rmse_loss=None,
                          sae_aux_loss=None,
                          var_expl=None,
                          kld=None,
                          perc_same_classification=None,
                          activation_similarity=None):
    '''A function for printing results and logging them to W&B. 

    Parameters:
    epoch_mode (str): The string 'train' or 'eval' to indicate whether
                         the results are from training or evaluation.
    use_sae (Bool): True or False
    batch (int): index of batch whose results are logged
    dead_neurons_steps (int): after how many batches/epochs we measure dead neurons
    '''
    if epoch_mode == 'train':
        epoch_or_batch = 'batch'
        step = batch
    elif epoch_mode == 'eval':
        epoch_or_batch = 'epoch'
        step = epoch
    else:
        raise ValueError("epoch_mode needs to be 'train' or 'eval'.")

    if epoch_mode == 'eval' or (epoch_mode == 'train' and step % 100 == 0):
        print("---------------------------------")
        print(f"Model loss: {model_loss:.4f} | Model accuracy: {accuracy:.4f}")
        if use_sae:
            print(f"KLD: {kld:.4f} | Perc same classifications: {perc_same_classification:.4f}")
            print(f"Loss difference: {loss_diff:.4f}")
        print("Variance explained", var_expl)
        print("Percentage of dead neurons", perc_dead_neurons)
    if wandb_status:
        wandb.log({f"{epoch_mode}/model loss": model_loss, f"{epoch_mode}/model accuracy": accuracy, f"{epoch_or_batch}": step}, commit=False)
        if use_sae:
            wandb.log({f"{epoch_mode}/KLD": kld, 
                       f"{epoch_mode}/Perc same classifications": perc_same_classification, 
                       f"{epoch_mode}/Loss difference": loss_diff,
                       f"{epoch_or_batch}": step}, commit=False)
            # wandb doesn't accept tuples as keys, so we convert them to strings
            perc_dead_neurons_wandb = {f"{epoch_mode}/Perc_of_dead_neurons_on_{epoch_mode}_data/{k[0]}_{k[1]}": v for k, v in perc_dead_neurons.items()}
            # merge two dictionaries and log them to W&B
            wandb.log({**perc_dead_neurons_wandb, f"{epoch_or_batch}": step}, commit=False) # overview of number of dead neurons for all layers
        
    # We show per model layer evaluation metrics
    for name in sparsity_dict.keys(): 
    # the names are the same for the polysemanticity and relative sparsity dictionaries
    # hence, it suffices to iterate over the keys of the sparsity dictionary
        model_key = name[1] # can be "original" (model), "sae" (encoder output), "modified" (model)

        #if name[0] in sae_layer or name[0] == 'fc2':
        if epoch_mode == 'eval' or (epoch_mode == 'train' and step % 100 == 0):
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
                print(f"SAE loss, layer {name[0]}: {sae_loss[name[0]]:.4f} | SAE rec. loss, layer {name[0]}: {sae_rec_loss[name[0]]:.4f} | SAE l1 loss, layer {name[0]}: {sae_l1_loss[name[0]]:.4f} | SAE aux loss, layer {name[0]}: {sae_aux_loss[name[0]]:.4f}")
            print(f"Sparsity, {model_key} layer {name[0]}: {sparsity_dict[name]:.3f}")
        if wandb_status:                    
            wandb.log({f"{epoch_mode}/Sparsity/{model_key}_layer_{name[0]} sparsity": sparsity_dict[name], f"{epoch_or_batch}":step}, commit=False)
            if use_sae and name[0] in sae_loss.keys():
                wandb.log({f"{epoch_mode}/SAE_loss/layer_{name[0]} SAE loss": sae_loss[name[0]], f"{epoch_or_batch}":step}, commit=False)
                wandb.log({f"{epoch_mode}/SAE_loss/layer_{name[0]} SAE rec. loss": sae_rec_loss[name[0]], f"{epoch_or_batch}":step}, commit=False)
                wandb.log({f"{epoch_mode}/SAE_loss/layer_{name[0]} SAE l1 loss": sae_l1_loss[name[0]], f"{epoch_or_batch}":step}, commit=False)
                wandb.log({f"{epoch_mode}/SAE_loss/layer_{name[0]} SAE nrmse loss": sae_nrmse_loss[name[0]], f"{epoch_or_batch}":step}, commit=False)
                wandb.log({f"{epoch_mode}/SAE_loss/layer_{name[0]} SAE rmse loss": sae_rmse_loss[name[0]], f"{epoch_or_batch}":step}, commit=False)
                wandb.log({f"{epoch_mode}/SAE_loss/layer_{name[0]} SAE aux loss": sae_aux_loss[name[0]], f"{epoch_or_batch}":step}, commit=False)
                wandb.log({f"{epoch_mode}/Variance_explained/layer_{name[0]}": var_expl[name[0]], f"{epoch_or_batch}":step}, commit=False)
                
                #aggregate_SAE_score = var_expl[name[0]]

             #if model_key == 'sae':
            #wandb.log({f"{epoch_mode}/Sparsity/{model_key}_layer_{name[0]} sparsity_1": sparsity_dict_1[name], f"{epoch_or_batch}":step}, commit=False)  
            # Optional metrics
            #if number_active_neurons is not None:
                #wandb.log({f"{epoch_mode}/Activation_of_neurons/{model_key}_layer_{name[0]} activated neurons": number_active_neurons[name][0], f"{epoch_or_batch}": step}, commit=False)            
                #wandb.log({f"{epoch_mode}/Number_of_neurons/{model_key}_layer_{name[0]}": number_active_neurons[name][1], f"{epoch_or_batch}": step}, commit=False)
                #wandb.log({f"{epoch_mode}/Activation_of_neurons/{model_key}_layer_{name[0]} dead neurons": number_dead_neurons[name], f"{epoch_or_batch}": step}, commit=False)
            if use_sae and activation_similarity is not None:
                wandb.log({f"{epoch_mode}/Feature_similarity_L2loss_between_modified_and_original_model/{model_key}_layer_{name[0]} mean": activation_similarity[name[0]][0], f"{epoch_or_batch}":step}, commit=False) 
                wandb.log({f"{epoch_mode}/Feature_similarity_L2loss_between_modified_and_original_model/{model_key}_layer_{name[0]} std": activation_similarity[name[0]][1], f"{epoch_or_batch}":step}, commit=False) 
            if mean_number_active_classes_per_neuron is not None:
                wandb.log({f"{epoch_mode}/Active_classes_per_neuron/{model_key}_layer_{name[0]} mean": mean_number_active_classes_per_neuron[name], f"{epoch_or_batch}":step}, commit=False)
                wandb.log({f"{epoch_mode}/Active_classes_per_neuron/{model_key}_layer_{name[0]} std": std_number_active_classes_per_neuron[name], f"{epoch_or_batch}":step}, commit=False)
            
    # we only log results at the end of an epoch, which is when epoch is not None (eval mode) or when batch is the last batch (train mode)
    if wandb_status:
    #    if (epoch is not None) or (batch is not None and batch == num_batches):
    #        print("Logging results to W&B...")
        wandb.log({}, commit=True) # commit the above logs

    
def feature_similarity(activations,activation_similarity,device):
    '''
    calculates the feature similarity between the modified and original for one batch
    '''
    unique_sae_layer = {key[0] for key in activations.keys()} # curly brackets denote a set --> only take unique values
    # alternatively: module_names = get_module_names(model)

    for name in unique_sae_layer:
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

def compute_sparsity(epoch_mode, 
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
        if epoch_mode == 'eval':
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
            if epoch_mode == 'eval':
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

                                
def get_top_k_samples(top_k_samples, batch_top_k_values, batch_top_k_indices, batch_filename_indices, eval_batch_idx, largest, k):
    '''
    top_k_samples = (previous_top_k_values of shape [k, #neurons], previous_top_k_indices of shape [k, #neurons], batch_size, filename_indices)
    --> We see that batch_size = top_k_samples[2]
    batch_top_k_values = top k values of current batch, shape [k, #neurons]
    batch_top_k_indices = top k indices of current batch, shape [k, #neurons]
    filename_indices = indices of the filenames in the dataset corresponding to the top k images, shape [k, #neurons]
    --> Want new top k values and indices incorporating the current and previous batches
    '''
    batch_size = top_k_samples[2]

    # we add batch_idx*batch_size to every entry in the indices matrix to get the 
    # index of the corresponding image in the dataset. For example, if the index in the
    # batch is 2 and the batch index is 4 and batch size is 64, then the index of this 
    # image in the dataset is 4*64 + 2= 258
    batch_top_k_indices += (eval_batch_idx - 1) * batch_size
    # we merge the previous top k values and the current top k values --> shape: [2*k, #neurons]
    # then we find the top k values within this matrix
    top_k_values_merged = torch.cat((top_k_samples[0], batch_top_k_values), dim=0)
    # we also merge the indices and the filename indices
    top_k_indices_merged = torch.cat((top_k_samples[1], batch_top_k_indices), dim=0)
    top_k_filename_indices_merged = torch.cat((top_k_samples[3], batch_filename_indices), dim=0)

    if top_k_values_merged.shape[0] < k:
        # f.e. if k = 200, but batch_size = 64, then after 2 batches, the top k values matrix has only 128 elements
        # and even after 3 batches, it has only 192 elements. Thus, we don't remove any values yet.
        top_k_values_merged_new = top_k_values_merged
        selected_indices = top_k_indices_merged
    else:
        # we find the top k values and indices of the merged top k values
        top_k_values_merged_new, top_k_indices_merged_new = torch.topk(top_k_values_merged, k=k, dim=0, largest=largest)
        # but top_k_indices_merged_new contains the indices of the top values within the top_k_values_merged
        # matrix, but we need to find the corresponding indices in top_k_indices_merged
        selected_indices = torch.gather(top_k_indices_merged, 0, top_k_indices_merged_new)
        top_k_filename_indices_merged = torch.gather(top_k_filename_indices_merged, 0, top_k_indices_merged_new)

    return (top_k_values_merged_new, selected_indices, batch_size, top_k_filename_indices_merged)


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
                       sae_layer=None, 
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
        file_path = get_file_path(folder_path, sae_layer, params, f'{model_key}_{layer_name}_{number_neurons}_top_{n**2}_samples.png')
        plt.savefig(file_path, bbox_inches='tight', pad_inches=0.1, dpi=300)
        print(f"Successfully stored top {n**2} samples for {number_neurons} neurons in {file_path}")
    plt.close()
    #'''


def show_top_k_samples_for_ie(val_dataloader, 
                       model_key, 
                       top_k_samples, 
                       small_k_samples, 
                       folder_path=None, 
                       sae_layer=None, 
                       params=None,
                       wandb_status=None,
                       dataset_name=None):

    # specify for which neurons we want to show the most and least activating samples
    layer_names = ["mixed3a", "mixed4b", "mixed4c", "mixed5a"]
    # we also have the 8 SAE errors in our circuit but we don't explain them here because they are high dimensional
    # so it doesnt make sense to find highly activating samples 
    number_neurons = 24
    neurons = (
        ("mixed3a", 118), 
        ("mixed3a", 426), 
        ("mixed3a", 605), 
        ("mixed3a", 1509),
        ("mixed4b", 948), 
        ("mixed4b", 1214), 
        ("mixed4b", 1264), 
        ("mixed4b", 1287),
        ("mixed4c", 802), 
        ("mixed4c", 918), 
        ("mixed4c", 1577), 
        ("mixed4c", 1847), 
        ("mixed4c", 1895),
        ("mixed5a", 111), 
        ("mixed5a", 564), 
        ("mixed5a", 1054), 
        ("mixed5a", 1424), 
        ("mixed5a", 1471), 
        ("mixed5a", 1606), 
        ("mixed5a", 1982), 
        ("mixed5a", 2092), 
        ("mixed5a", 2569), 
        ("mixed5a", 2731), 
        ("mixed5a", 2830)   
    )

    top_index_matrix = {}
    small_index_matrix = {}
    top_value_matrix = {}
    small_value_matrix = {}

    # matrix of shape [k=n**2, num_neurons] containing the indices of the top k samples for each neuron
    for name in layer_names:
        top_index_matrix[name] = top_k_samples[(name, "sae")][1]
        small_index_matrix[name] = small_k_samples[(name, "sae")][1]
        top_value_matrix[name] = top_k_samples[(name, "sae")][0]
        small_value_matrix[name] = small_k_samples[(name, "sae")][0]

    # based on the given number_neurons we find number of rows and columns
    # such that the number of rows and columns is as close as possible (approximately)
    # so that the figure doesn't become too wide or too high    
    num_cols, num_rows = rows_cols(number_neurons)

    fig = plt.figure(figsize=(18,9))#figsize=(num_rows*9, num_cols*18))
    outer_grid = fig.add_gridspec(num_rows, num_cols, wspace=0.1, hspace=0.3)
    fig.suptitle(f'{number_neurons} most and least activating samples for {number_neurons} neurons', fontsize=10)

    for i in range(num_rows):
        for j in range(num_cols):
            neuron = neurons[i * num_cols + j] # tuple: (layer_name, neuron_idx)
            layer_name = neuron[0]
            neuron_idx = neuron[1]
            
            if i * num_cols + j > number_neurons:
                break
            
            # indices of the top k samples for the neuron with index neuron_idx
            top_indices = top_index_matrix[layer_name][:,neuron_idx]
            small_indices = small_index_matrix[layer_name][:,neuron_idx]
            top_values = top_value_matrix[layer_name][:,neuron_idx]
            small_values = small_value_matrix[layer_name][:,neuron_idx]

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
        file_path = get_file_path(folder_path, sae_layer, params, f'{model_key}_{layer_name}_{number_neurons}_top_{n**2}_samples.png')
        plt.savefig(file_path, bbox_inches='tight', pad_inches=0.1, dpi=300)
        print(f"Successfully stored top {n**2} samples for {number_neurons} neurons in {file_path}")
    plt.close()
    #'''

def process_sae_layers_list(sae_layers, original_model, training):
    '''
    Turns the string 'fc1&fc2&&fc3&fc4' into the list ['fc1&fc2&fc3', 'fc1&fc2&fc3&fc4']
    --> For each element of the list (separated by underscores), the code will train an SAE on the last element of the list
        (if training = "True" and original_model = "False") and use pretrained SAEs on the preceding layers --> here:
        fc1&fc2&fc3 --> Train SAE on fc3 and use pretrained SAEs on fc1, and fc1_fc2 (i.e. on fc2 but given that SAE on fc1 is already trained)
        fc1&fc2&fc3&fc4 --> Train SAE on fc4 and use pretrained SAEs on fc1, fc1&fc2, fc1&fc2&fc3 
    '''
    pretrained_sae_layers_string, train_sae_layers_string = sae_layers.split("&&")
    # Split the sub_string into a list based on '_'
    #pretrained_sae_layers_list = pretrained_sae_layers_string.split("&")
    train_sae_layers_list = train_sae_layers_string.split("&")

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
                name = pretrained_sae_layers_string + "&" + train_sae_layers_list[i]
            sae_layers_list.append(name)
            pretrained_sae_layers_string = name
    else:
        sae_layers_list = [sae_layers]  
    return sae_layers_list

def activation_histograms(activations, folder_path, sae_layer, params, wandb_status, targets=None):
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
        file_path = get_file_path(folder_path=folder_path, sae_layer=sae_layer, params=params, file_name=f'{name}_{key[0]}_{key[1]}.png')
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

def plot_lucent_explanations(model, sae_layer, params, folder_path, wandb_status, num_units):
    folder_path = os.path.join(folder_path, 'lucent_explanations')
    os.makedirs(folder_path, exist_ok=True)
    layer_list = []
    if sae_layer == "None__": # if we use the original model
        print("Not computing maximally activating explanations for the original model. If you want to do so, please specify layers manually in the plot_lucent_explanations function.")
    else: # if we use SAE
        # count the number of non-empty strings in sae_layer.split("_")
        if sum([1 for name in sae_layer.split("_") if name != ""]) > 1:
            print("Should we compute maximally activating explanations for all SAE layers? As of right now, nothing is computed")
        else:
            for name in sae_layer.split("_"):
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
            img = render.render_vis(model, "{}:{}".format(layer_name, i), show_image=False, preprocess="torchvision")[0]
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
        file_path = get_file_path(folder_path=folder_path, sae_layer=sae_layer, params=params, file_name=f'lucent_explanations_{layer_name}_for_{num_units}_neurons.png')
        plt.savefig(file_path, dpi=300)
        plt.close()
        print(f"Successfully stored maximally activating explanations of layer {layer_name} in {file_path}")

def update_histogram(histogram_info, name, model_key, output, device, output_2=None):
    # we unpack the values contained in the dictionary
    histogram_matrix = histogram_info[(name,model_key)][0]
    top_values = histogram_info[(name,model_key)][1]
    small_values = histogram_info[(name,model_key)][2]
    neuron_indices = histogram_info[(name,model_key)][3]
    num_bins = histogram_matrix.shape[0]
    num_units = histogram_matrix.shape[1]

    # we get the activations of the current batch 
    if output_2 is not None and model_key=='sae': # we get the pre-relu encoder output
        activations = output_2
    elif output_2 is None and model_key!='sae':
        activations = output
    else:
        raise ValueError(f"model_key is {model_key} but output_2 is {output_2}")

    # we only consider the specified units
    activations = activations[:,neuron_indices]
    activations = activations.to(device)
    
    # we compute the histogram of the activations
    for unit in range(num_units):
        top_value = top_values[unit].item() # these are invariant over the batches and hence we will hae the same bins over all batches
        small_value = small_values[unit].item() 
        histogram_matrix[:, unit] += torch.histc(activations[:,unit], bins=num_bins, min=small_value, max=top_value)

    # we write the updated histogram_matrix back into the dictionary
    histogram_info[(name,model_key)] = (histogram_matrix, top_values, small_values, neuron_indices)
    return histogram_info


def activation_histograms_2(histogram_info, folder_path, sae_layer, params, wandb_status, epoch):
    name = "activation_histograms"
    folder_path = os.path.join(folder_path, name)
    os.makedirs(folder_path, exist_ok=True)
    for key, v in histogram_info.items():
        histogram_matrix = v[0]
        top_values = v[1]
        small_values = v[2]
        neuron_indices = v[3]
        num_bins = histogram_matrix.shape[0] 
        num_units = histogram_matrix.shape[1]
        cols, rows = rows_cols(num_units)
        fig = plt.figure(figsize=(18,12))
        plt.suptitle(f"Histograms of neuron activations, {key}, epoch {epoch}")
        for i in range(num_units):
            edges = torch.linspace(small_values[i], top_values[i], num_bins + 1)
            plt.subplot(rows, cols, i+1)
            plt.stairs(values=histogram_matrix[:, i].cpu().numpy(), edges=edges, fill=True)
            plt.xlabel('Activation value')
            plt.ylabel('No. of samples')
            plt.title(f'Neuron {neuron_indices[i]}')
        fig.tight_layout(pad=1.0)
        if wandb_status:
            wandb.log({f"eval/{name}/{key[0]}_{key[1]}":wandb.Image(plt)})
        # store the figure also if we use the cluster because resolution with W&B might not be high enough
        file_path = get_file_path(folder_path=folder_path, sae_layer=sae_layer, params=params, file_name=f'{name}_{key[0]}_{key[1]}_epoch_{epoch}.png')
        plt.savefig(file_path, dpi=300)
        plt.close()
        print(f"Successfully stored {name} of layer {key[0]}, model {key[1]} in {file_path}")

def average_over_W_H(output, output_2):
    # if we consider conv layers, we sometimes want to consider 
    # the average over width and height
    if len(output.shape) == 4:
        # we compute the average activation for each channel
        output = torch.mean(output, dim=(2,3)) # the desired output has shape (b c) as we take the average over h and w
    # if the output has 2 dimensions, we just keep it as it is          
    # --> modified output has shape [b, #units]
        
    if output_2 is not None:
        if len(output_2.shape) == 4:
            output_2 = torch.mean(output_2, dim=(2,3))
        # else we just keep output_2 as it is

    return output, output_2

def variance_explained(output, decoder_output):
    if len(output.shape) == 4:
        # take variance over width and height
        var = torch.var(output, dim=(2,3))
        # take average over channels and batches
        var = torch.mean(var)
        if len(decoder_output.shape) != 4:
            raise ValueError(f"Decoder output has unexpected shape {len(decoder_output.shape)}.")
        mod_var = torch.mean(torch.var(decoder_output, dim=(2,3)))
    elif len(output.shape) == 2: # [B, #units]
        var = torch.var(output, dim=1)
        var = torch.mean(var)
        if len(decoder_output.shape) != 2: 
            raise ValueError(f"Decoder output has unexpected shape {len(decoder_output.shape)}.")
        mod_var = torch.mean(torch.var(decoder_output, dim=1))
    else:
        raise ValueError(f"Output has unexpected shape {len(output.shape)}.")
    
    return 1 - mod_var/var

def measure_inactive_units(output, expansion_factor):
    bool_output = output == 0 # --> True if 0, False otherwise
    # using a small threshold: bool_output = output < 1e-2
    bool_output = bool_output.bool()

    # For each sample in the batch, we compute whether a unit is inactive or active (True if inactive, False if active)
    # the tensor sample_inactive_units has shape [BS, #units]
    # - conv case: unit = channel --> [BS, C]; MLP case: unit = neuron --> [BS, #neurons]
    if len(output.shape) == 4:
        # output has shape [BS, C, H, W]
        # a channel is inactive if all pixels are zero
        sample_inactive_units = torch.all(torch.all(bool_output, dim=3), dim=2) # for some reason dim = (2,3) doesn't work so we do it sequentially
        # for measuring sparsity pixel-wise, use the below
        ###sample_inactive_units = rearrange(bool_output, 'b c h w -> b (c h w)')
    elif len(output.shape) == 2:
        # output has shape [BS, #neurons] already
        sample_inactive_units = bool_output
    else:
        raise ValueError(f"Output has unexpected shape {len(output.shape)}.")
    number_of_units = sample_inactive_units.size(1)
    
    # if a unit is inactive (True) across the batch (dim=0), then the unit is dead (across this batch)    
    batch_dead_units = torch.all(sample_inactive_units, dim=0)
    # shape: [#units]

    # sample_inactive_units is of shape [BS, #units] with True if a unit is inactive for a sample
    # we then convert to float, i.e., True --> 1 and take mean for each unit, then we do 1 - mean to get the frequency of active units
    neuron_activity_frequency = 1 - torch.mean(sample_inactive_units.float(), dim=0)

    # sparsity is computed sample wise
    # --> we count the number of inactive channels of a sample, i.e., across the units (dim=1) 
    number_inactive_units_per_sample = torch.sum(sample_inactive_units, dim=1) # shape: [BS]
    number_active_units_per_sample = number_of_units - number_inactive_units_per_sample
    sparsity_per_sample = number_active_units_per_sample / (number_of_units / expansion_factor) # shape: [BS]
    # now we compute the average sparsity across the batch
    batch_sparsity = torch.mean(sparsity_per_sample).item()
       
    return batch_dead_units, batch_sparsity, neuron_activity_frequency

# adapted from: https://github.com/zimmerrol/machine-interpretability/blob/c271c7a526c857ff91d7634da8b5abaadd62cf19/stimuli_generation/sg_utils.py#L701
def get_label_translator(directory_path):
    """
    old labels:
        https://raw.githubusercontent.com/rgeirhos/lucent/dev/lucent/modelzoo/misc/old_imagenet_labels.txt
    new labels:
        https://raw.githubusercontent.com/conan7882/GoogLeNet-Inception/master/data/imageNetLabel.txt
    """
    if directory_path.startswith('/lustre'):
        old_labels_path = os.path.join(directory_path, 'dataloaders/old_imagenet_labels.txt')
        new_labels_path = os.path.join(directory_path, 'dataloaders/imagenet_labels.txt')
    elif directory_path.startswith('C:'):
        old_labels_path = os.path.join(directory_path, 'dataloaders\\old_imagenet_labels.txt')
        new_labels_path = os.path.join(directory_path, 'dataloaders\\imagenet_labels.txt')
    else:
        raise ValueError(f"Unexpected directory path: {directory_path}")

    with open(old_labels_path, "r", encoding="utf-8") as fhandle:
        old_imagenet_labels_data = fhandle.read()
    with open(new_labels_path, "r", encoding="utf-8") as fhandle:
        new_imagenet_labels_data = fhandle.read()

    # maps a class index to wordnet-id in old convention
    old_imagenet_labels_map = {}
    #old_imagenet_cid_to_wid = {}
    for cid, l in enumerate(old_imagenet_labels_data.strip().split("\n")):
        wid = l.split(" ")[0].strip()
        old_imagenet_labels_map[wid] = cid
        #old_imagenet_cid_to_wid[cid] = wid

    # maps a class index to wordnet-id in new convention
    new_imagenet_labels_map = {}
    for cid, l in enumerate(new_imagenet_labels_data.strip().split("\n")):
        wid = l.split(" ")[0].strip()
        new_imagenet_labels_map[cid] = wid

    def remap_torch_to_tf_labels(y):
        """Map PyTorch-style ImageNet labels to old convention used by GoogLeNet/InceptionV1."""
        res = []
        for yi in y.cpu().numpy():
            zi = None
            wid = new_imagenet_labels_map[yi]
            if wid in old_imagenet_labels_map:
                zi = old_imagenet_labels_map[wid]
                res.append(zi)

            else:
                raise ValueError(f"Unknown class {yi}/{wid}.")

        return torch.tensor(res).to(y.device) + 1
    
    return remap_torch_to_tf_labels


def process_batch(batch, 
                  directory_path=None,
                  epoch_mode=None, 
                  filename_to_idx=None):
    if isinstance(batch, (list, tuple)) and len(batch) == 2:
        inputs, targets = batch
        # dummy filenames, just to return something, but we don't use them as we only
        # use the imagenet filenames for computing the MIS
        filename_indices = torch.arange(len(inputs))
    elif isinstance(batch, dict) and len(batch) == 2 and list(batch.keys())[0] == "image" and list(batch.keys())[1] == "label":
        # this format holds for the tiny imagenet dataset
        inputs, targets = batch["image"], batch["label"]
        filename_indices = torch.arange(len(inputs))
    elif isinstance(batch, list) and len(batch) == 3:
        # this format holds for imagenet
        inputs = batch[0] # shape is [batch_size, 3, 229, 229]
        targets = batch[1] # tensor of indices
        filenames = batch[2] # list of filenames

        # we assume here that we consider the imagenet dataset, if we're given this data format!!!
        ###label_translator = get_label_translator(directory_path)
        # maps Pytorch imagenet labels to labelling convention used by InceptionV1
        ###targets = label_translator(targets) # f.e. tensor([349, 898, 201,...]), i.e., tensor of label indices
        # this translation is necessary because the InceptionV1 model return output indices in a different convention
        # hence, for computing the accuracy f.e. we need to translate the Pytorch indices to the InceptionV1 indices

        # write the filenames to a txt file, one filename per line
        # can impose a special requirement (such as specifying one class)
        '''
        txt_file = "/lustre/home/jtoussaint/master_thesis/imagenet_filenames.txt"
        with open(txt_file, "a") as f:
            for filename in filenames:
                #if "n02007558" in filename: # write the filenames of all flamingo images
                f.write(filename + "\n")
        '''

        # if we only consider the flamingo images
        #'''
        indices = [index for index, filename in enumerate(filenames) if "n02007558" in filename] # flamingo indices (99% eval acc)
        #indices = [index for index, filename in enumerate(filenames) if "n01756291" in filename] # sidewinder indices (40% eval acc)
        #indices = [index for index, filename in enumerate(filenames) if "n02788148" in filename] # bannister indices (73% eval acc)
        inputs = inputs[indices]
        targets = targets[indices]
        #'''

        if epoch_mode is not None:
            if epoch_mode == "mis" or epoch_mode == "compute_max_min_samples":
                # Get the indices of the filenames in the text file
                # get(s,-1) returns -1 if s is not in the dictionary...
                filename_indices = []
                for s in filenames: 
                    if s not in filename_to_idx:
                        raise ValueError(f"Filename {s} not in filename_to_idx.")
                    else: 
                        filename_indices.append(filename_to_idx.get(s))
                    
                # convert to tensor
                filename_indices = torch.tensor(filename_indices, dtype=torch.long)
                filename_indices = filename_indices[indices] # we only consider the indices of the flamingo images
                # torch.all(targets == filename_indices)) --> False for every batch --> we should use filename_indices 
                # Instead of computing these filename_indices, we could perhaps also use the target/class indices, but we later translate them
                # to match the target index convention used by Inception. Thus, it would make things more complicated.
            else:
                filename_indices = torch.arange(len(inputs))
        else: 
            filename_indices = torch.arange(len(inputs))

        # write indices to a txt file
        '''
        with open('/lustre/home/jtoussaint/master_thesis/dataloaders/imagenet_train_val_indices.txt', 'a') as file:
            # Write each index on a new line
            for label in batch[1]:
                file.write(str(label.item()) + '\n')
        '''

        # we write the imagenet filenames to a txt file
        # we use filenames for computing the MIS. We only compute MIS for imagenet.
        # this txt file has to be created only once, hence it is commented here
        '''
        with open('/lustre/home/jtoussaint/master_thesis/dataloaders/imagenet_train_train_filenames.txt', 'a') as file:
            # Write each filename on a new line
            for filename in batch[2]: # batch[2] is a list of filenames
                file.write(filename + '\n')  # Add newline after each filename
        '''
    else:
        raise ValueError(f"Unexpected data format from dataloader: {type(batch)} with length {len(batch)}, in particular: {batch}")
    return inputs, targets, filename_indices

def get_string_to_idx_dict(filename):
    '''
    Input is a text file with one filename (imagefile) per line, f.e., dataloaders/imagenet_train_train_filenames.txt
    which contains the filenames of all images in the Imagenet training set.
    The index is created by the order of the filenames in the text file.
    '''
    with open(filename, "r") as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines] # Clean up newlines at the end of each line
    filename_to_index = {filename: idx for idx, filename in enumerate(lines)}
    index_to_filename = {idx: filename for filename, idx in filename_to_index.items()}
    return filename_to_index, index_to_filename


def compute_mis(evaluation_results_folder_path, 
                params_string, 
                params_string_1, 
                model_key, 
                layer_name, 
                idx_to_filename, 
                n_mis, 
                epoch, 
                device,
                sae_lambda_sparse, 
                sae_expansion_factor,
                sae_batch_size,
                sae_optimizer_name,
                sae_learning_rate):
    
    folder_path = os.path.join(evaluation_results_folder_path, 'filename_indices')
    file_path = get_file_path(folder_path=folder_path,
                            sae_layer=model_key + '_' + layer_name,
                            params=params_string,
                            file_name=f'max_min_filename_indices_epoch_{epoch}.pt')
    data = torch.load(file_path, map_location=device)
    max_filename_indices = data['max_filename_indices']
    min_filename_indices = data['min_filename_indices']

    similarity_function = "dreamsim" 
    similarity_function_args = (
        "/is/cluster/fast/rzimmermann/interpretability-comparison-data/dependencies/dreamsim_features_imagenet_train.pkl",
        "/is/cluster/fast/rzimmermann/interpretability-comparison-data/dependencies/dreamsim_logistic_regression.hard_coded_sklearn.pkl" #"../../stimuli_generation/dreamsim_logistic_regression.hard_coded.pkl"
    ) 

    mis_list = []
    mis_confidence_list = []
    idx_list = []

    # we want to compute the MIS for each unit in the layer                
    for unit_idx in range(max_filename_indices.shape[1]):
    
        max_filename_indices_unit = max_filename_indices[:,unit_idx]
        #print(max_filename_indices_unit)
        min_filename_indices_unit = min_filename_indices[:,unit_idx]
        max_filenames = [idx_to_filename[idx.item()] for idx in max_filename_indices_unit]
        min_filenames = [idx_to_filename[idx.item()] for idx in min_filename_indices_unit]
        '''
        make_fair_batches says that it takes a list of filenames sorted by ascending activation score but in fact
        if one looks at sg_utils.extract_stimuli_for_layer_units_from_dataframe line 916 & 917, we see that the input 
        is min_exemplars = min_queries + min_refs; max_exemplars = max_refs + max_queries (i.e. the lists are added as such: [1,2] + [3,4] = [1,2,3,4])
        But the order according to activation values is: min_refs < min_queries < max_queries < max_refs
        So the input is not by "ascending activation score", while inside of all of these lists however, the order is indeed by 
        ascending activation score.
        Apart from the wrong description of make_fair_batches, the code is correct. The requirement is simply that the query images
        come last (for min_exemplars we apply reverse=True). It might not make much of a difference though if the query images 
        are chosen to be as such: min_queries < min_refs < max_refs < max_queries
        '''
        # query images come last
        max_queries_filenames = max_filenames[:n_mis] # get first n_mis (# tasks) filenames 
        max_refs_filenames = max_filenames[n_mis:] # remaining filenames
        min_queries_filenames = min_filenames[-n_mis:] # last n_mis filenames
        min_refs_filenames = min_filenames[:-n_mis] # remaining filenames

        max_filenames = max_refs_filenames + max_queries_filenames
        min_filenames = min_queries_filenames + min_refs_filenames # we apply reverse=True below                    
        
        max_lists = sg_utils.make_fair_batches(max_filenames, n_mis)
        min_lists = sg_utils.make_fair_batches(min_filenames, n_mis, reverse=True)
        # from: sg_utils.extract_stimuli_for_layer_units_from_dataframe
        #print("max_lists:", max_lists) 
        #print("min_lists:", min_lists)
        batch_filenames = [maxs + mins for mins, maxs in zip(min_lists, max_lists)]
        
        compute_machine_interpretability_score = mis_utils.prepare_machine_interpretability_score(
                similarity_function, similarity_function_args
            )
        #print("batch filenames:", batch_filenames)
        
        mis, mis_confidence = compute_machine_interpretability_score(batch_filenames, include_individual_scores=False) 
        print(f"Unit index: {unit_idx}, MIS: {mis}, MIS confidence: {mis_confidence}")
        
        mis_list.append(mis)
        mis_confidence_list.append(mis_confidence)
        idx_list.append(unit_idx)

        #if unit_idx == 5:
        #    break

    median_mis = np.median(mis_confidence_list)
    average_mis = np.mean(mis_confidence_list)
    print(f"Median MIS for layer {layer_name}: {median_mis}")
    print(f"Average MIS for layer {layer_name}: {average_mis}")

    layer_name_list = [layer_name] * len(idx_list)
    model_key_list = [model_key] * len(idx_list)
    #'''
    df = pd.DataFrame({'unit_idx': idx_list, 'MIS': mis_list, 'MIS_confidence': mis_confidence_list, 'layer_name': layer_name_list, 'model_key': model_key_list})

    folder_path_df = os.path.join(evaluation_results_folder_path, 'MIS')
    if not os.path.exists(folder_path_df):
        os.makedirs(folder_path_df)
    file_path = get_file_path(folder_path=folder_path_df,
                            sae_layer=model_key + '_' + layer_name,
                            params=params_string,
                            file_name=f'mis_epoch_{epoch}.csv')
    df.to_csv(file_path)

    # we store the mis median and average in the sae results file
    # we make certain assumptions here, f.e., that this file already exists with all other values filled in
    # this usually happens since we first train and eval the SAE before computing the MIS
    store_sae_eval_results(evaluation_results_folder_path, 
                        layer_name, 
                        params_string_1, 
                        epoch,
                        sae_lambda_sparse, 
                        sae_expansion_factor,
                        sae_batch_size,
                        sae_optimizer_name,
                        sae_learning_rate,
                        median_mis=median_mis)
    #'''
   

def collect_images_from_subfolders_and_delete_subfolders(parent_dir):
    '''
    Move all files from all subfolders to the parent directory and delete subfolders.
    '''
    # Walk through all subdirectories and files
    for root, dirs, files in os.walk(parent_dir, topdown=False):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            dest_path = os.path.join(parent_dir, file_name)
                
            # Move the file to the parent directory
            shutil.move(file_path, dest_path)
        
        # Remove the now-empty subfolders
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            os.rmdir(dir_path)
    
    print("Images moved and subfolders deleted successfully.")


def show_imagenet_images(unit_idx, directory_path, params_string, epoch="None", evaluation_results_folder_path=None, model_key=None, layer_name=None, device=None, max_filename_indices=None, descending=False):
    '''
    This function shows imagenet images given a list of filename indices, i.e., we can visualize the
    max and min samples corresponding to specific units.    
    '''
    if max_filename_indices is None:
        folder_path = os.path.join(evaluation_results_folder_path, 'filename_indices')
        file_path = get_file_path(folder_path=folder_path,
                                sae_layer=model_key + '_' + layer_name,
                                params=params_string,
                                file_name=f'max_min_filename_indices_epoch_{epoch}.pt')
        data = torch.load(file_path, map_location=device)
        max_filename_indices = data['max_filename_indices']
        min_filename_indices = data['min_filename_indices']

    #n = 25 # show only the first n images

    filename_txt = os.path.join(directory_path, 'dataloaders/imagenet_train_filenames.txt')
    filename_to_idx, idx_to_filename = get_string_to_idx_dict(filename_txt)

    #print("Max filename indices shape:", max_filename_indices.shape)
    # shape: [k, #units] (where k=200 here, top k samples)

    #if descending:
    #    max_filename_indices_unit = max_filename_indices[:n, unit_idx]
    #else:
    #    max_filename_indices_unit = max_filename_indices[-n:, unit_idx]
    max_filename_indices_unit = max_filename_indices
    #max_filenames = [idx_to_filename[idx.item()] for idx in max_filename_indices_unit]
    # for downloading all flamingo images
    max_filenames = [idx_to_filename[idx] for idx in max_filename_indices_unit]
    max_filenames = [i + ".JPEG.jpg" for i in max_filenames]

    print(max_filenames)

    #min_filename_indices_unit = min_filename_indices[:n, unit_idx]
    #min_filenames = [idx_to_filename[idx.item()] for idx in min_filename_indices_unit]
    #min_filenames = [i + ".JPEG.jpg" for i in min_filenames]
    min_filenames = []

    print(min_filenames)

    output_dir = os.path.join(directory_path, 'imagenet_images', params_string + "_epoch_" + epoch + "_unit" + str(unit_idx))
    output_dir_max = os.path.join(output_dir, "maximally_activating")
    output_dir_min = os.path.join(output_dir, "minimally_activating")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir_max, exist_ok=True)
    os.makedirs(output_dir_min, exist_ok=True)
    datadir = "/fast/rzimmermann/ImageNet2012-webdataset"
    extracted_files = []

    for i in range(0, 147):
        tar_filename = os.path.join(datadir, "imagenet-train-{:06d}.tar".format(i))
        print("Checking tar file:", tar_filename)
        with tarfile.open(tar_filename, 'r') as tar:
            for filenames, name in zip([max_filenames, min_filenames],["max", "min"]):
                print("Checking ", name, " filenames")
                for filename in filenames:
                    if filename in tar.getnames():
                        # we get the actual filename without the folders in front, f.e.: "train/n03000134/n03000134_3248.JPEG.jpg" --> "n03000134_3248.JPEG.jpg"
                        base_filename = os.path.basename(filename)
                        if name == "max":
                            output_dir = output_dir_max
                            output_file_path = os.path.join(output_dir, base_filename)
                        else:
                            output_dir = output_dir_min
                            output_file_path = os.path.join(output_dir, base_filename)
                        if not os.path.exists(output_file_path):
                            print("Extracting file:", base_filename)
                            tar.extract(filename, output_dir)
                            # --> extracts the file to f.e. output_dir/train/n03000134/n03000134_3248.JPEG.jpg
                            extracted_files.append(base_filename)
                        else:
                            print("File already exists:", base_filename)            
    print(extracted_files)

    # we put all files into the respective output_dir directly deleting all subfolders
    collect_images_from_subfolders_and_delete_subfolders(output_dir_max)
    collect_images_from_subfolders_and_delete_subfolders(output_dir_min)


def sae_inference_and_loss(sae_model_name, sae_model, sae_criterion_name, output, sae_criterion, sae_lambda_sparse):
    # the output of the specified layer of the original model is the input of the SAE
    # if the output has 4 dimensions, we flatten it to 2 dimensions along the scheme: (BS, C, H, W) -> (BS*W*H, C)
    sae_input, transformed = reshape_tensor(output)                     

    if sae_model_name == "sae_mlp":
        encoder_output, decoder_output, encoder_output_prerelu = sae_model(sae_input) 
    elif sae_model_name == "gated_sae":
        encoder_output, decoder_output, relu_pi_gate, via_gate = sae_model(sae_input)
        encoder_output_prerelu = None
    else:
        raise ValueError(f"Unknown SAE model name {sae_model_name}.")
    
    if transformed:
        encoder_output = rearrange(encoder_output, '(b h w) c -> b c h w', b=output.size(0), h=output.size(2), w=output.size(3))
        if encoder_output_prerelu is not None:
            encoder_output_prerelu = rearrange(encoder_output_prerelu, '(b h w) c -> b c h w', b=output.size(0), h=output.size(2), w=output.size(3))
        # note that the encoder_output(_prerelu) has c*k channels, where k is the expansion factor  
            
    if sae_model_name == "sae_mlp" and sae_criterion_name == "sae_loss":
        rec_loss, l1_loss, nrmse_loss, rmse_loss = sae_criterion(encoder_output, decoder_output, sae_input) # the sae inputs are the targets
        aux_loss = torch.tensor(0)
        loss = rec_loss + sae_lambda_sparse * l1_loss
    elif sae_model_name == "gated_sae" and sae_criterion_name == "gated_sae_loss":
        rec_loss, l1_loss, nrmse_loss, rmse_loss, aux_loss = sae_criterion(relu_pi_gate, via_gate, decoder_output, sae_input)
        loss = rec_loss + sae_lambda_sparse * l1_loss + aux_loss
    else:
        raise ValueError(f"Unknown combination of SAE criterion name {sae_criterion_name} and SAE model name {sae_model_name}.")
    
    if transformed:                
        decoder_output = rearrange(decoder_output, '(b h w) c -> b c h w', b=output.size(0), h=output.size(2), w=output.size(3))
        assert decoder_output.shape == output.shape
        sae_input = rearrange(sae_input, '(b h w) c -> b c h w', b=output.size(0), h=output.size(2), w=output.size(3))  

    return loss, rec_loss, l1_loss, nrmse_loss, rmse_loss, aux_loss, encoder_output, encoder_output_prerelu, decoder_output



def compute_ie(encoder_outputs, encoder_output_average, sae_error_average, decoder_weight_matrix, sae_errors, number_sae_channels, gradients):
    '''   
    SAE FEATURE NODES

    Compute the indirect effect of SAE encoder output nodes, which correspond to a conv channel.
    Let N = number of samples (f.e. batch size), C = number of channels, H = height, W = width, K = expansion factor of SAE
    We have: 
    - gradients of model loss wrt layer output x of shape [N, C, H, W] 
    - mean SAE encoder output on Imagenet of shape [C*K, H, W]
    - SAE encoder outputs on all circuit images of shape [N, C*K, H, W]
    - SAE decoder weight matrix of shape [C, C*K]

    The IE formula that we use assumes encoder output of dimension [C*K], i.e., for one sample out of B*H*W samples 
    i.e., we treat the convolutional layers as if they were linear layers. Hence, we do the same rearrangement/transformation
    as before: [B, C, H, W] --> [B*H*W, C]. Overall, we average the IE over B,H,W and store the IE for each layer and SAE unit i.

    SAE ERROR NODES

    For the SAE error nodes, we have:
    - gradients of model loss wrt layer output x of shape [N, C, H, W]
    - mean SAE error on Imagenet of shape [C, H, W]
    - SAE errors on all circuit images of shape [N, C, H, W]

    In this case, we take the average over all dimensions to obtain one IE value for the whole SAE error node.
    '''
    batch_size = gradients.size(0)

    encoder_output_average = torch.unsqueeze(encoder_output_average, 0) # add a new dimension at the beginning -> shape: [1, C*K, H, W]
    encoder_output_average = encoder_output_average.repeat(batch_size, 1, 1, 1) # repeat the tensor N times along the new dimension -> shape: [N, C*K, H, W]
    encoder_output_average = rearrange(encoder_output_average, 'b c h w -> (b h w) c') # unfold the tensor to shape [NHW, C*K]

    sae_error_average = torch.unsqueeze(sae_error_average, 0) # add a new dimension at the beginning -> shape: [1, C, H, W]
    sae_error_average = sae_error_average.repeat(batch_size, 1, 1, 1) # repeat the tensor N times along the new dimension -> shape: [N, C, H, W]
    sae_error_average = rearrange(sae_error_average, 'b c h w -> (b h w) c') # unfold the tensor to shape [NHW, C]
    
    encoder_outputs = rearrange(encoder_outputs, 'b c h w -> (b h w) c') # shape [NHW, C*K]
    gradients = rearrange(gradients, 'b c h w -> (b h w) c') # shape [NHW, C], 
    # recall that these are the gradients with respect to the layer output in the original model, hence we have C channels
    sae_errors = rearrange(sae_errors, 'b c h w -> (b h w) c') # shape [NHW, C]

    IE_sae_features = []

    ############## COMPUTE IE FOR EACH SAE NODE / CHANNEL ##############
    for channel_idx in range(number_sae_channels):
        feature_direction_i = decoder_weight_matrix[:, channel_idx] # v_i, shape [C]
        feature_direction_i = torch.unsqueeze(feature_direction_i, 1) # shape [C, 1]
        encoder_output_i = encoder_outputs[:, channel_idx] # shape [NHW]
        encoder_output_i = torch.unsqueeze(encoder_output_i, 1) # shape [NHW, 1]
        encoder_output_average_i = encoder_output_average[:, channel_idx] # shape [NHW]
        encoder_output_average_i = torch.unsqueeze(encoder_output_average_i, 1) # shape [NHW, 1]

        # by the chain rule
        grad = gradients @ feature_direction_i # matrix multiplication: [NHW, C] @ [C, 1] = [NHW, 1]

        '''
        In the basic case (IE formula (3) presented in Marks paper), without conv layer (H=W=1) and one sample (N=1), we have [1, C] @ [C, 1] = [1],
        where C is the number of units. Now, for N samples, i.e., if we are given tensors of shape [N, C] and [C, N], we want to get 10 scalar values
        by multiplying each row of the first tensor by the corresponding column of the second tensor. These 10 values correspond to the 10 values that
        one obtains by doing matrix multiplication of those tensors and then considering the diagonal elements (which are the result of multipling the 
        i-th row by the i-th column). Hence, we do:
        [NHW, C] @ [NHW, C].T = [NHW, C] @ [C, NHW] = [NHW, NHW] and then take the diagonal elements --> [NHW]. 
        However, computing a matrix product of shape [200k,200k] is quite expensive, so we can be more efficient using torch.einsum
        With torch.einsum: 'ik,kj->ij' is matrix multiplication and 'ii->i' gives the diagonal elements. Hence, we do 'nc,cn->n'  
        Then, we take the mean.

        For the SAE error nodes we have C = number of channels in corresponding layer of original model. However, for the SAE feature nodes, we have C=1. 
        In this case, the procedure can be simplified (even though I just use the same procedure as for the SAE error nodes, because it is more general).
        If C=1, we don't need to worry about taking the dot product of two vectors, but instead, for each N,H,W we can simply do scalar multiplication of
        gradient and (encoder_output_average_i - encoder_output_i). Doing this for all N,H,W, we can just do pointwise multiplication: 
        [NHW, 1] * [NHW, 1] = [NHW, 1], and then we can take the mean over the first dimension.

        Code: 
        IE = grad * (encoder_output_average_i - encoder_output_i)
        IE = torch.mean(IE, dim=0).item() # shape: scalar
        '''
        ie = torch.einsum('nc,cn->n', grad, (encoder_output_average_i - encoder_output_i).T) # shape: [NHW]
        # we take the mean over NHW, i.e., over all "samples"
        ie = torch.mean(ie) # shape: scalar
        IE_sae_features.append(ie) 
    # once the loop is done, IE_sae_features is a list of scalars of length = number of channels in SAE

    ############## COMPUTE IE FOR SAE ERROR ##############
    ie = torch.einsum('nc,cn->n', gradients, (sae_error_average - sae_errors).T) # shape: [NHW]
    IE_sae_error = torch.mean(ie) # shape: scalar
            
    return torch.tensor(IE_sae_features), IE_sae_error


def compute_ie_all_channels(sae_errors, sae_error_average, model_gradients, batch_size):
    '''
    Return only one value for all channels together.

    SAE ERROR NODES

    For the SAE error nodes, we have:
    - gradients of model loss wrt layer output x of shape [N, C, H, W]
    - mean SAE error on Imagenet of shape [C, H, W]
    - SAE errors on all circuit images of shape [N, C, H, W]

    In this case, we take the average over all dimensions to obtain one IE value for the whole SAE error node.
    '''
    # batch_size = model_gradients.size(0)

    sae_error_average = torch.unsqueeze(sae_error_average, 0) # add a new dimension at the beginning -> shape: [1, C, H, W]
    sae_error_average = sae_error_average.repeat(batch_size, 1, 1, 1) # repeat the tensor N times along the new dimension -> shape: [N, C, H, W]
    sae_error_average = rearrange(sae_error_average, 'b c h w -> (b h w) c') # unfold the tensor to shape [NHW, C]
    
    # if model_gradients has 4 dimensions reshape, else keep as they are
    #if len(model_gradients.size()) == 4:
    model_gradients = rearrange(model_gradients, 'b c h w -> (b h w) c') # shape [NHW, C]
    sae_errors = rearrange(sae_errors, 'b c h w -> (b h w) c') # shape [NHW, C]

    ie = torch.einsum('nc,cn->n', model_gradients, (sae_error_average - sae_errors).T) # shape: [NHW]
    ie = torch.abs(ie) # we are interested in the absolute IE
    ie_sae_error = torch.mean(ie) # shape: scalar

    return ie_sae_error
            


def compute_ie_channel_wise(encoder_outputs, encoder_output_average, encoder_gradients, batch_sizes):
    '''   
    Return a distinct value for each channel.

    SAE FEATURE NODES

    Compute the indirect effect of SAE encoder output nodes, which correspond to a conv channel.
    Let N = number of samples (f.e. batch size), C = number of channels, H = height, W = width, K = expansion factor of SAE
    We have: 
    - gradients of model loss wrt layer output x of shape [NHW, C*K] 
    - mean SAE encoder output on Imagenet of shape [C*K, H, W]
    - SAE encoder outputs on all circuit images of shape [NHW, C*K]
    - SAE decoder weight matrix of shape [C, C*K]

    The IE formula that we use assumes encoder output of dimension [C*K], i.e., for one sample out of B*H*W samples 
    i.e., we treat the convolutional layers as if they were linear layers. Hence, we do the same rearrangement/transformation
    as before: [B, C, H, W] --> [B*H*W, C]. Overall, we average the IE over B,H,W and store the IE for each layer and SAE unit i.
    '''
    encoder_output_average = reshape_encoder_output_average(encoder_output_average, batch_sizes) # shape: [1, NHW, C*K]

    # encoder_outputs already in shape [NHW, C*K]
    encoder_outputs = torch.unsqueeze(encoder_outputs, 0) # shape: [1, NHW, C*K]

    # encoder_gradients already in shape [NHW, C*K]
    encoder_gradients = torch.unsqueeze(encoder_gradients, 1) # shape: [NHW, 1, C*K]

    # we do the same as below, except that the channel dimension is just trailing behind not having any particular effect
    ie = torch.einsum('nic,inc->nc', encoder_gradients, encoder_output_average - encoder_outputs)  # Shape: [NHW, C*K]

    ie = torch.abs(ie) # we are interested in the absolute IE
    
    ie_sae_features = torch.mean(ie, dim=0)  # Shape: [C*K]
    # --> for each channel we its IE value

    ''' # same as above but with a for-loop (longer, less elegant)
    IE_sae_features = []
    number_sae_channels = encoder_outputs.size(1) # C*K
    for channel_idx in range(number_sae_channels):
        encoder_output_i = encoder_outputs[:, channel_idx] # shape [NHW]
        encoder_output_i = torch.unsqueeze(encoder_output_i, 1) # shape [NHW, 1]
        encoder_output_average_i = encoder_output_average[:, channel_idx] # shape [NHW]
        encoder_output_average_i = torch.unsqueeze(encoder_output_average_i, 1) # shape [NHW, 1]

        grad = encoder_gradients[:, channel_idx] # shape [NHW]
        grad = torch.unsqueeze(grad, 1) # shape [NHW, 1]
       
        ie = torch.einsum('nc,cn->n', grad, (encoder_output_average_i - encoder_output_i).T) # shape: [NHW]
        # we take the mean over NHW, i.e., over all "samples"
        ie = torch.mean(ie) # shape: scalar
        IE_sae_features.append(ie) 
    # once the loop is done, IE_sae_features is a list of scalars of length = number of channels in SAE
    ie_sae_features = torch.tensor(IE_sae_features)
    '''

    return ie_sae_features

def get_specific_sae_params(layer_name,
                           sae_model_name,
                           model_params_temp,
                           sae_optimizer_name):
    
    ############################# SPECIFY FOR EACH LAYER WHICH PARAMETERS TO USE #############################
    sae_batch_size = '256'
    # so far, for all layers apart from mixed3a, I just chose some random values to see if the pipeline works

    if layer_name == "mixed3a" or layer_name == "inception3a":
        dead_neurons_steps = 626
        sae_checkpoint_epoch = 7
        sae_learning_rate = 0.001
        sae_lambda_sparse = 5.0
        sae_expansion_factor = 8
    elif layer_name == "mixed3b" or layer_name == "inception3b":
        dead_neurons_steps = 625
        sae_checkpoint_epoch = 6
        sae_learning_rate = 0.001
        sae_lambda_sparse = 0.1
        sae_expansion_factor = 4
    elif layer_name == "mixed4a" or layer_name == "inception4a":
        dead_neurons_steps = 625
        sae_checkpoint_epoch = 6
        sae_learning_rate = 0.001
        sae_lambda_sparse = 0.1
        sae_expansion_factor = 4
    elif layer_name == "mixed4b" or layer_name == "inception4b":
        dead_neurons_steps = 625
        sae_checkpoint_epoch = 6
        sae_learning_rate = 0.001
        sae_lambda_sparse = 0.1
        sae_expansion_factor = 4
    elif layer_name == "mixed4c" or layer_name == "inception4c":
        dead_neurons_steps = 625
        sae_checkpoint_epoch = 5
        sae_learning_rate = 0.001
        sae_lambda_sparse = 0.1
        sae_expansion_factor = 4
    elif layer_name == "mixed4d" or layer_name == "inception4d":
        dead_neurons_steps = 625
        sae_checkpoint_epoch = 7
        sae_learning_rate = 0.001
        sae_lambda_sparse = 0.1
        sae_expansion_factor = 4
    elif layer_name == "mixed4e" or layer_name == "inception4e":
        dead_neurons_steps = 625 # previously, mistakenly I set it to 199, even though it should be 300, also for 5a and 5b
        sae_checkpoint_epoch = 9
        sae_learning_rate = 0.001
        sae_lambda_sparse = 0.1
        sae_expansion_factor = 4
    elif layer_name == "mixed5a" or layer_name == "inception5a":
        dead_neurons_steps = 625
        sae_checkpoint_epoch = 5
        sae_learning_rate = 0.001
        sae_lambda_sparse = 0.1
        sae_expansion_factor = 4
    elif layer_name == "mixed5b" or layer_name == "inception5b":
        dead_neurons_steps = 625
        sae_checkpoint_epoch = 12
        sae_learning_rate = 0.001
        sae_lambda_sparse = 0.1
        sae_expansion_factor = 4

    ############################# DON'T CHANGE ANYTHING BELOW #############################
    # NOTE: the sae_params below is different than the one defined in other places because I omit the sae_epochs parameter!
    sae_params = {'sae_model_name': sae_model_name, 'learning_rate': sae_learning_rate, 'batch_size': sae_batch_size, 'optimizer': sae_optimizer_name, 'expansion_factor': sae_expansion_factor, 
                            'lambda_sparse': sae_lambda_sparse, 'dead_neurons_steps': dead_neurons_steps}
    sae_params_temp = {k: str(v) for k, v in sae_params.items()}
    sae_params_string_ie = '_'.join(model_params_temp.values()) + "_" + "_".join(sae_params_temp.values()) + f"_sae_checkpoint_epoch_{sae_checkpoint_epoch}"

    # this is an irrelevant parameter but it is inluded in the MIS name anyways (should be removed in the future)
    if layer_name == "mixed3a" or layer_name == "inception3a":
        sae_epoch = "11"
    else:
        sae_epoch = "13"
    sae_params_temp.pop('sae_model_name')
    sae_params_string_mis = '_'.join(model_params_temp.values()) + f"_{sae_model_name}_{sae_epoch}_" + "_".join(sae_params_temp.values()) + f"_mis_epoch_{sae_checkpoint_epoch}"

    return sae_params_string_ie, sae_checkpoint_epoch, sae_expansion_factor, sae_params_string_mis, dead_neurons_steps
  


def get_specific_sae_model(layer_name,
                           layer_size,
                           sae_model_name,
                           sae_weights_folder_path,
                           model_params_temp,
                           device,
                           sae_optimizer_name):
    params_string_sae_checkpoint, sae_checkpoint_epoch, sae_expansion_factor, _, _ = get_specific_sae_params(layer_name, sae_model_name, model_params_temp, sae_optimizer_name)

    sae_model = load_model(sae_model_name, img_size=layer_size, expansion_factor=sae_expansion_factor)
    if sae_checkpoint_epoch > 0:
        file_path = get_file_path(sae_weights_folder_path, layer_name, params=params_string_sae_checkpoint, file_name= '.pth')
        checkpoint = torch.load(file_path, map_location=device)
        state_dict = checkpoint['model_state_dict']
        sae_model.load_state_dict(state_dict)
        print(f"Use SAE on layer {layer_name} from epoch {sae_checkpoint_epoch}")
    sae_model = sae_model.to(device)
    
    sae_model.eval()
    for param in sae_model.parameters():
        param.requires_grad = False
    
    return sae_model, params_string_sae_checkpoint, int(sae_expansion_factor)


def reshape_tensor(tensor):
    if len(tensor.shape) == 4:
        return rearrange(tensor, 'b c h w -> (b h w) c'), True
    else: # if the output has 2 dimensions, we just keep it as it is          
        return tensor, False  
    
def reshape_encoder_output_average(tensor, batch_size):
    tensor = torch.unsqueeze(tensor, 0) # add a new dimension at the beginning -> shape: [1, C*K, H, W]
    tensor = tensor.repeat(batch_size, 1, 1, 1) # repeat the tensor N times along the new dimension -> shape: [N, C*K, H, W]
    tensor = rearrange(tensor, 'b c h w -> (b h w) c') # unfold the tensor to shape [NHW, C*K]
    tensor = torch.unsqueeze(tensor, 1) # shape: [NHW, 1, C*K]
    tensor = tensor.permute(1, 0, 2) # shape: [1, NHW, C*K]
    return tensor



def apply_sae(sae, model_output, nodes=None, ablation=None):
    b = model_output.size(0)
    h = model_output.size(2)
    w = model_output.size(3)
    model_output, transformed = reshape_tensor(model_output)

    # compute SAE output
    encoder_output, decoder_output, _ = sae(model_output)

    if nodes is not None:
        # adjust the encoder_output as desired
        # nodes is a tensor of shape [C*K] with True/False, where True means to keep the original value
        # and False means to ablate it to the value specified in ablation
        # if ablation is encoder output average it has shape [C*K, H, W]
        # encoder_output has shape [NHW, C*K] --> rearrange to [N, C*K, H, W] 
        new_encoder_output = encoder_output.clone()
        new_encoder_output = rearrange(new_encoder_output, '(b h w) c -> b c h w', b=b, h=h, w=w) # shape: [N, C*K, H, W]
        # for each sample (refers to ... below), we set all H & W values of the nodes/channels which are False to the ablation value
        new_encoder_output[..., ~nodes, :, :] = ablation[~nodes, :, :] 
        # rearrange back to [NHW, C*K]
        new_encoder_output = rearrange(new_encoder_output, 'b c h w -> (b h w) c')
        new_decoder_output = sae.decoder(new_encoder_output)
    else:
        new_decoder_output = decoder_output

    if transformed: # required for computing sae_error in right dimensions for passing to model
        # doing rearrange and then .save() (or vice versa) leads to an error! Hence, we don't rearrange 
        # the encoder_output. But anyways, for computing the ie we use the encoder output in its current
        # form. Or we could just rearrange outside of the trace context. We rearrange the two quantities
        # below because they need to be passed as new layer output in the right format.
        decoder_output = rearrange(decoder_output, '(b h w) c -> b c h w', b=b, h=h, w=w)
        new_decoder_output = rearrange(new_decoder_output, '(b h w) c -> b c h w', b=b, h=h, w=w)
        model_output = rearrange(model_output, '(b h w) c -> b c h w', b=b, h=h, w=w)

    return encoder_output, decoder_output, new_decoder_output