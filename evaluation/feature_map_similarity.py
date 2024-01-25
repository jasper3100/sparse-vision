import torch
import os

from utils_feature_map import load_feature_map
    
def intermediate_feature_maps_similarity(module_names, original_activations_folder_path, adjusted_activations_folder_path):
    '''
    Calculate the similarity between the intermediate feature maps of the original and adjusted model
    We only need to compare the feature maps after the first SAE, because before that they are
    identical.
    '''
    similarity_mean_list = []
    similarity_std_list = []
    L2_dist_mean_list = []
    L2_dist_std_list = []

    for name in module_names:
        original_activations_file_path = os.path.join(original_activations_folder_path, f'{name}_activations.h5')
        adjusted_activations_file_path = os.path.join(adjusted_activations_folder_path, f'{name}_activations.h5')

        original_feature_map = load_feature_map(original_activations_file_path)
        adjusted_feature_map = load_feature_map(adjusted_activations_file_path)
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
        print(
            f"Layer: {module_names[i]} | Cosine similarity mean: {similarity_mean_list[i]} +/- {similarity_std_list[i]} | L2 distance mean: {L2_dist_mean_list[i]} +/- {L2_dist_std_list[i]}")