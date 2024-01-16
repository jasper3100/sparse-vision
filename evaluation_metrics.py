import torch.nn.functional as F
import torch
import h5py
import os

from main import original_model_output_file_path, adjusted_model_output_file_path, activations_folder_path
from model import weights

'''
Contents of this file:
- load the output of original and modified model
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



def ce(original_output, adjusted_output):
    '''
    We don't want the model's output to change after applying the SAE. The cross-entropy is 
    suitable for comparing probability outputs. Hence, we want the cross-entropy between 
    the original model's output and the output of the modified model to be small.
    '''
    # cross_entropy(input, target), where target consists of probabilities
    original_output = F.softmax(original_output, dim=1) 
    return F.cross_entropy(adjusted_output, original_output)



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



def intermediate_feature_maps_similarity():
    '''
    Calculate the similarity between the intermediate feature maps of the original and adjusted model
    We only need to compare the feature maps after the first SAE, because before that they are
    identical.
    '''
    return 1

file_path = os.path.join(activations_folder_path, 'model.fc_intermediate_activations.h5') 
with h5py.File(file_path, 'r') as h5_file:
    original_output_2 = torch.from_numpy(h5_file['data'][:])


original_output, adjusted_output = load_output(original_model_output_file_path, adjusted_model_output_file_path)
ce = ce(original_output, adjusted_output)
percentage_same_classification = percentage_same_classification(original_output, adjusted_output)
print(f"Cross-entropy between original model's output and modified model's output: {ce:.4f}")
print(f"Percentage of samples with same classification (between original and modified model): {percentage_same_classification:.1f}%")
print_classifications(original_output, adjusted_output, weights)

# check if original_output and original_output_2 are the same
print(original_output == original_output_2)
print(original_output[0])
print(original_output_2[0])