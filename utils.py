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
import wandb
import csv
import pandas as pd

from dataloaders.tiny_imagenet import TinyImageNetDataset, TinyImageNetPaths
from models.custom_mlp_1 import CustomMLP1
from models.sae_conv import SaeConv
from models.sae_mlp import SaeMLP
from losses.sparse_loss import SparseLoss
from supplementary.dataset_stats import print_dataset_stats

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
    elif dataset_name == 'mnist':   
        return (1, 28, 28)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
def get_file_path(folder_path=None, layer_names=None, params=None, file_name=None, params2=None):
    '''
    layer_names expects a list of layer(s), params and params2 expect a dictionary of parameters
    '''
    if folder_path is not None:
        os.makedirs(folder_path, exist_ok=True) # create the folder

    if layer_names is None:
        pass
    else:
        layer_names = '_'.join(layer_names)

    if params is not None and params2 is None:
        params_values = [str(value) for value in params.values()]
        file_name = f'{layer_names}_{"_".join(params_values)}_{file_name}'
    elif params is not None and params2 is not None:
        params_values = [str(value) for value in params.values()]
        params2_values = [str(value) for value in params2.values()]
        file_name = f'{layer_names}_{"_".join(params_values)}_{"_".join(params2_values)}_{file_name}'
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
                        sae_expansion_factor=None, # only needed for SAE models
                        layer_names=None,# only needed for SAE models, which are trained on activations of a specific layer
                        params=None):  
    model = load_model(model_name, img_size, sae_expansion_factor)
    file_path = get_file_path(folder_path, layer_names, params, 'model_weights.pth')
    model.load_state_dict(torch.load(file_path))
    model.eval()
    return model

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

def load_data(directory_path, dataset_name, batch_size):
    if dataset_name == 'tiny_imagenet':
        root_dir='datasets/tiny-imagenet-200'
        # if root_dir does not exist, download the dataset
        root_dir=os.path.join(directory_path, root_dir)
        download = not os.path.exists(root_dir)
        if not os.path.exists(root_dir):
            os.makedirs(root_dir, exist_ok=True)

        # Data shuffling should be turned off here so that the activations that we store in the model without SAE
        # are in the same order as the activations that we store in the model with SAE
        train_dataset = TinyImageNetDataset(root_dir, mode='train', preload=False, download=download)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        val_dataset = TinyImageNetDataset(root_dir, mode='val', preload=False, download=download)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

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

        transform = transforms.Compose([
            transforms.ToTensor(),
            # Cifar images have values in [0,1]. We scale them to be in [-1,1].
            # Some thoughts on why [-1,1] is better than [0,1]:
            # https://datascience.stackexchange.com/questions/54296/should-input-images-be-normalized-to-1-to-1-or-0-to-1
            # We can verify that using the values (0.5,0.5,0.5),(0.5,0.5,0.5) will yield data in [-1,1]:
            # verify using: print_dataset_stats(train_dataset)
            # to call this function, in this script, do:
            # directory_path = r'C:\Users\Jasper\Downloads\Master thesis\Code'
            # load_data(directory_path=directory_path, dataset_name='cifar_10', batch_size=32)
            # Theoretical explanation: [0,1] --> (0 - 0.5)/0.5 = -1 and (1 - 0.5)/0.5 = 1
            # instead, using the mean and std instead, will yield normalized data (mean=0, std=1) but not in [-1,1]
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
    
    elif dataset_name == 'mnist':
        root_dir='datasets/mnist'
        # if root_dir does not exist, download the dataset
        root_dir=os.path.join(directory_path, root_dir)
        download = not os.path.exists(root_dir)
        if not os.path.exists(root_dir):
            os.makedirs(root_dir, exist_ok=True)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,),(0.5,))
        ])

        train_dataset = torchvision.datasets.MNIST(root_dir, train=True, download=download, transform=transform)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        val_dataset = torchvision.datasets.MNIST(root_dir, train=False, download=download, transform=transform)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        category_names = train_dataset.classes    
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

def show_classification_with_images(train_dataloader,
                                    class_names, 
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

    input_images, target_ids = next(iter(train_dataloader))  
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
        img = input_images[i] / 2 + 0.5 # unnormalize the image
        npimg = img.cpu().numpy()
        axes[i + 1].imshow(np.transpose(npimg, (1, 2, 0)))

        if output_2 is not None:
            title = f'{class_names[target_ids[i]]}\n{predicted_classes[i]} ({100*scores[i].item():.1f}%)\n{predicted_classes_2[i]} ({100*scores_2[i].item():.1f}%)'
        else:
            title = f'{class_names[target_ids[i]]}\n{predicted_classes[i]} ({100*scores[i].item():.1f}%)'

        axes[i + 1].set_title(title, fontsize=8)
        axes[i + 1].axis('off')

    plt.subplots_adjust(wspace=0.5)  # Adjust space between images
    plt.savefig(file_path)
    plt.close()
    #plt.show()
    print(f"Successfully stored classification visualizations in {file_path}")

def log_image_table(train_dataloader,
                    class_names, 
                    model=None,
                    device=None,
                    output=None,
                    output_2=None):
    n = 10  # show only the first n images, 
    # for showing all images in the batch use len(predicted_classes)

    images, target_ids = next(iter(train_dataloader))  
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

    number_bins = 20
    bin_width = num_classes / number_bins
    # Set bin edges, we start from a small constant > 0 because we create a separate bin
    # for zero values (dead neurons) to distinguish between dead neurons and ultra sparse neurons
    # For example, if num_classes = 10 --> bin_width = 0.5 --> bin edges: [epsilon, 0.5), [0.5, 1.0), ...
    bins1 = np.array([np.finfo(float).eps, bin_width])
    bins2 = np.arange(bin_width, num_classes+0.0001, bin_width) 
    # adding some small value to end point because otherwise np.arange does not include the end point
    bins = np.concatenate((bins1, bins2))

    for name in number_active_classes_per_neuron.keys():
        if name[0] in layer_names or name[0] == 'fc3':
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

            axes[row, col].hist(number_active_classes_per_neuron[name].cpu().numpy(), bins=bins, color='blue')
            axes[row, col].set_title(f'Layer: {name[0]}, Model: {name[1]}')
            axes[row, col].set_xlabel('Number of active classes')
            axes[row, col].set_ylabel('Number of neurons')
            # set a custom tick label of the tick at x=np.finfo(float).eps
            x_ticks = axes[row, col].get_xticks()
            axes[row, col].set_xticks(np.append(x_ticks, (0, -0.5))) # add a tick at x=0 and x=-0.5
            axes[row, col].set_xticklabels([str(int(x)) for x in x_ticks] + [r' $\epsilon$', '0']) # label at x=0 is epsilon and at x=-0.5 is 0

            # Add a red bar for values equal to zero (dead neurons)
            dead_neurons = (number_active_classes_per_neuron[name].cpu().numpy() == 0).sum()
            axes[row, col].bar(-0.25, dead_neurons, color='red', width=0.5, label=f'{dead_neurons} dead neurons')
            axes[row, col].legend(loc='upper right')

    plt.subplots_adjust(wspace=0.7)  # Adjust space between images
    
    if wandb_status:
        wandb.log({"active_classes_per_neuron":wandb.Image(plt)})
    else:
        os.makedirs(folder_path, exist_ok=True)
        file_path = get_file_path(folder_path, layer_names, params, 'active_classes_per_neuron.png')
        plt.savefig(file_path)
        plt.close()
        print(f"Successfully stored active classes per neuron plot in {file_path}")


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
    inactive_neurons = x < threshold # get the indices of the neurons below the threshold
    
    inactive_neurons = torch.prod(inactive_neurons, dim=0) # we collapse the batch size dimension. In particular,
    # inactive_neurons is of shape [batch size (samples in batch), number of neurons in layer]. We multiply the 
    # rows element-wise, so that we get a tensor of shape [number of neurons in layer], where each
    # element is True if the neuron is dead in all batches, and False otherwise.

    # the below quantity is summed over all samples in one batch
    number_active_neurons = (x >= threshold).sum().item()

    # get the total number of neurons in the layer, i.e. the second dimension of x
    number_total_neurons = x.shape[1]

    return inactive_neurons, number_active_neurons, number_total_neurons


def compute_sparsity(activating_neurons, total_neurons, expansion_factor=None):
    if expansion_factor is not None:
        # If we are in the augmented layer (i.e. the output of SAE encoder), then 
        # we compute the sparsity relative to the size of the original layer
        number_of_neurons_in_original_layer = total_neurons / expansion_factor
        return 1 - (activating_neurons / number_of_neurons_in_original_layer) 
    else:
        return 1 - (activating_neurons / total_neurons) 

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
                            train_rec_loss, 
                            train_scaled_l1_loss, 
                            relative_sparsity):
    # remove lambda_sparse and expansion_factor from params, because we want a uniform file name 
    # for all lambda_sparse and expansion_factor values
    file_path = get_file_path(folder_path=folder_path,
                            layer_names=layer_names,
                            params=params,
                            file_name='sae_eval_results.csv')
    file_exists = os.path.exists(file_path)
    columns = ["lambda_sparse", "expansion_factor", "rec_loss", "l1_loss", "rel_sparsity"]

    # We get the relative sparsity of the encoder output
    for name in layer_names:
        rel_sparsity = relative_sparsity[name, "sae"]

    if not file_exists:
        with open(file_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=columns)
            writer.writeheader()
            writer.writerow({columns[0]: lambda_sparse, 
                             columns[1]: expansion_factor, 
                             columns[2]: train_rec_loss, 
                             columns[3]: train_scaled_l1_loss, 
                             columns[4]: rel_sparsity})
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
                        row["rec_loss"] = str(train_rec_loss)
                        row["l1_loss"] = str(train_scaled_l1_loss)
                        row["rel_sparsity"] = str(rel_sparsity)
                        break
            else:
                # If the combination doesn't exist, add a new row
                rows.append({"lambda_sparse": str(lambda_sparse), 
                             "expansion_factor": str(expansion_factor), 
                             "rec_loss": str(train_rec_loss), 
                             "l1_loss": str(train_scaled_l1_loss),
                             "rel_sparsity": str(rel_sparsity)})

        # Write the updated data back to the CSV file
        with open(file_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=columns)
            writer.writerows(rows)
    print(f"Successfully stored SAE eval results with lamda_sparse and expansion_factor in {file_path}")

def get_folder_paths(directory_path, model_name, dataset_name, sae_model_name):
    model_weights_folder_path = os.path.join(directory_path, 'model_weights', model_name, dataset_name)
    sae_weights_folder_path = os.path.join(directory_path, 'model_weights', sae_model_name, dataset_name)
    activations_folder_path = os.path.join(directory_path, 'feature_maps', model_name, dataset_name)
    evaluation_results_folder_path = os.path.join(directory_path, 'evaluation_results', model_name, dataset_name)
    return model_weights_folder_path, sae_weights_folder_path, activations_folder_path, evaluation_results_folder_path


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
    counting_matrix = torch.zeros(d, num_classes)
    above_threshold = above_threshold.to(counting_matrix.dtype)  # Convert to the same type
    counting_matrix.index_add_(1, target, above_threshold.t())
    # The code is equivalent to the below for loop, which is too slow though but easier to understand
    '''
    for i in range(n):
        for j in range(d):
            counting_matrix[j, target[i]] += encoder_output[i, j] >= activation_threshold
    '''

    # If we just want to get the number of classes that each neuron is active without caring about which classes and how
    # often a neuron was active on it, we can do the following: For each row i (each dimension i of the activations), 
    # we count the number of distinct positive integers, i.e., the number of classes that have an activation above the threshold
    # .bool() turns every non-zero element into a 1, and every zero into a 0
    # torch.sum(counting_matrix.bool(), dim=1)

    return counting_matrix

def compute_number_dead_neurons(dead_neurons):
    '''
    dead_neurons is a dictionary of the form {(model_type,layer_name): [True,False,False,...]},
    where 'True' means that the neuron is dead
    '''
    number_dead_neurons = {}
    for key, tensor in dead_neurons.items():
        number_dead_neurons[key] = tensor.sum().item()
    return number_dead_neurons