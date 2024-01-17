import torch.nn.functional as F
import torch
import h5py
import os

from main import original_activations_folder_path, adjusted_activations_folder_path, layer_name
from model import model, weights
from auxiliary_functions import names_of_main_modules_and_specified_layer

'''
Contents of this file:
- load feature map of original and modified model
- cross-entropy loss between original model's output and modified model's output
- print final classification of both models
- percentage of samples with same classification as before
'''

def load_feature_map(original_activation_file_path, adjusted_activation_file_path):
    # Load the original model's feature map
    with h5py.File(original_activation_file_path, 'r') as h5_file:
        original_output = torch.from_numpy(h5_file['data'][:])

    # Load the modified model's feature map
    with h5py.File(adjusted_activation_file_path, 'r') as h5_file:
        adjusted_output = torch.from_numpy(h5_file['data'][:])
    
    return original_output, adjusted_output

def load_feature_map_last_layer(model, 
                                original_activations_folder_path, 
                                adjusted_activations_folder_path):
    # get the names of the main modules of the model and include layer_name
  
    module_names = names_of_main_modules_and_specified_layer(model, layer_name)
    # the output layer is the second last layer, because layer_name is the last
    output_layer = module_names[-2]

    original_output_file_path = os.path.join(original_activations_folder_path, f'{output_layer}_activations.h5')
    adjusted_output_file_path = os.path.join(adjusted_activations_folder_path, f'{output_layer}_activations.h5')
    
    original_output, adjusted_output = load_feature_map(original_output_file_path, 
                                                        adjusted_output_file_path)
    return original_output, adjusted_output



def ce(original_feature_map, adjusted_feature_map):
    '''
    We don't want the model's output to change after applying the SAE. The cross-entropy is 
    suitable for comparing probability outputs. Hence, we want the cross-entropy between 
    the original model's output and the output of the modified model to be small.
    '''
    # cross_entropy(input, target), where target consists of probabilities
    original_feature_map = F.softmax(original_feature_map, dim=1) 
    return F.cross_entropy(adjusted_feature_map, original_feature_map)



def print_classifications(original_output, adjusted_output, weights):
    original_prob = F.softmax(original_output, dim=1) 
    adjusted_prob = F.softmax(adjusted_output, dim=1) 
    original_score, original_class_ids = original_output.max(dim=1)
    adjusted_score, adjusted_class_ids = adjusted_output.max(dim=1)
    original_category_names = [weights.meta["categories"][index] for index in original_class_ids]
    adjusted_category_names = [weights.meta["categories"][index] for index in adjusted_class_ids]
    
    # for each sample, print category name and score next to each other
    for i in range(original_output.size(0)):
        print(f"Sample {i+1}: {original_category_names[i]}: {original_score[i]:.1f}% | {adjusted_category_names[i]}: {adjusted_score[i]:.1f}%")

    

def percentage_same_classification(original_output, adjusted_output):
    '''
    Calculate percentage of samples with same classification as before
    '''
    original_prob = F.softmax(original_output, dim=1) 
    adjusted_prob = F.softmax(adjusted_output, dim=1) 
    original_class_ids = original_output.argmax(dim=1)
    adjusted_class_ids = adjusted_output.argmax(dim=1)
    return (original_class_ids == adjusted_class_ids).sum().item() / original_class_ids.size(0)



def intermediate_feature_maps_similarity(model, layer_name):
    '''
    Calculate the similarity between the intermediate feature maps of the original and adjusted model
    We only need to compare the feature maps after the first SAE, because before that they are
    identical.
    '''
    # get the names of the main modules of the model and include layer_name
    module_names = names_of_main_modules_and_specified_layer(model, layer_name)

    # store layer name, similarity mean, similarity std, L2 distance mean, L2 distance std in a dictionary
    # and print it
    similarity_mean_list = []
    similarity_std_list = []
    L2_dist_mean_list = []
    L2_dist_std_list = []

    for name in module_names:
        original_activations_file_path = os.path.join(original_activations_folder_path, f'{name}_activations.h5')
        adjusted_activations_file_path = os.path.join(adjusted_activations_folder_path, f'{name}_activations.h5')

        original_feature_map, adjusted_feature_map = load_feature_map(original_activations_file_path, 
                                                                      adjusted_activations_file_path)

        # flatten feature maps: we keep the batch dimension (dim=0) and flatten the rest
        # because cosine similarity can only be computed between two 1D tensors
        original_feature_map = torch.flatten(original_feature_map, start_dim=1, end_dim=-1)
        adjusted_feature_map = torch.flatten(adjusted_feature_map, start_dim=1, end_dim=-1)

        # calculate cosine similarity, keeping the batch dimension
        similarity = torch.cosine_similarity(original_feature_map, adjusted_feature_map, dim=1)
        # calculate mean similarity over the batch
        similarity_mean = round(similarity.mean().item(), 2)
        # calculate standard deviation over the batch; here might be some samples for which
        # the similarity is much lower than for others --> want to capture this
        similarity_std = round(similarity.std().item(), 2)

        # calculate euclidean distance between feature maps
        L2_dist = torch.linalg.norm(original_feature_map - adjusted_feature_map, dim=1)
        L2_dist_mean = round(L2_dist.mean().item(), 2)
        L2_dist_std = round(L2_dist.std().item(), 2)

        similarity_mean_list.append(similarity_mean)
        similarity_std_list.append(similarity_std)
        L2_dist_mean_list.append(L2_dist_mean)
        L2_dist_std_list.append(L2_dist_std)

    for i in range(len(module_names)):
        print(f"Layer: {module_names[i]} | Cosine similarity mean: {similarity_mean_list[i]} +/- {similarity_std_list[i]} | L2 distance mean: {L2_dist_mean_list[i]} +/- {L2_dist_std_list[i]}")
    
    

original_output, adjusted_output = load_feature_map_last_layer(model,
                                                                original_activations_folder_path,
                                                                adjusted_activations_folder_path)
ce = ce(original_output, adjusted_output)
print(f"Cross-entropy between original model's output and modified model's output: {ce:.4f}")

percentage_same_classification = percentage_same_classification(original_output, adjusted_output)
print(f"Percentage of samples with same classification (between original and modified model): {percentage_same_classification:.1f}%")

print_classifications(original_output, adjusted_output, weights)

intermediate_feature_maps_similarity(model, layer_name)