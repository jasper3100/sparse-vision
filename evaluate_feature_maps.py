import os
import torch
import torch.nn.functional as F

from utils import load_feature_map, get_classifications, show_classification_with_images, get_module_names

def intermediate_feature_maps_similarity(module_names, 
                                         original_activations_folder_path, 
                                         adjusted_activations_folder_path,
                                         L2_dist=True,
                                         cosine_similarity=False):
    '''
    Calculate the similarity between the intermediate feature maps of the original and adjusted model
    We only need to compare the feature maps after the first SAE, because before that they are
    identical.
    '''
    if cosine_similarity:
        similarity_mean_list = []
        similarity_std_list = []
    if L2_dist:
        L2_dist_mean_list = []
        L2_dist_std_list = []

    for name in module_names:
        original_activations_file_path = os.path.join(original_activations_folder_path, f'{name}_activations.h5')
        adjusted_activations_file_path = os.path.join(adjusted_activations_folder_path, f'{name}_activations.h5')

        original_feature_map = load_feature_map(original_activations_file_path)
        adjusted_feature_map = load_feature_map(adjusted_activations_file_path)

        if cosine_similarity:
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
            similarity_mean_list.append(similarity_mean)
            similarity_std_list.append(similarity_std)

        if L2_dist:
            # calculate euclidean distance between feature maps
            L2_dist = torch.linalg.norm(original_feature_map - adjusted_feature_map, dim=1)
            L2_dist_mean = round(L2_dist.mean().item(), 2)
            L2_dist_std = round(L2_dist.std().item(), 2)        
            L2_dist_mean_list.append(L2_dist_mean)
            L2_dist_std_list.append(L2_dist_std)

    for i in range(len(module_names)):
        print(
            f"Layer: {module_names[i]} | Cosine similarity mean: {similarity_mean_list[i]} +/- {similarity_std_list[i]} | L2 distance mean: {L2_dist_mean_list[i]} +/- {L2_dist_std_list[i]}")

def get_feature_map_last_layer(module_names, folder_path):
    # the output layer is the last layer
    output_layer = module_names[-1]
    file_path = os.path.join(folder_path, f'{output_layer}_activations.h5')
    return load_feature_map(file_path)

def print_percentage_same_classification(original_output, adjusted_output):
    '''
    Print percentage of samples with same classification as before
    '''
    _, _, original_class_ids = get_classifications(original_output)
    _, _, adjusted_class_ids = get_classifications(adjusted_output)
    percentage_same_classification = (original_class_ids == adjusted_class_ids).sum().item() / original_class_ids.size(0)
    print(f"Percentage of samples with the same classification (between original and modified model): {percentage_same_classification:.7f}%")

def compute_ce(feature_map_1, feature_map_2):
    # cross_entropy(input, target), where target consists of probabilities
    feature_map_1 = F.softmax(feature_map_1, dim=1)
    #feature_map_1 = feature_map_1.to(torch.float64)
    #feature_map_2 = feature_map_2.to(torch.float64)
    return F.cross_entropy(feature_map_2, feature_map_1)

def get_accuracy(output, target):
    _, _, class_ids = get_classifications(output)
    return (class_ids == target).sum().item() / target.size(0)

def evaluate_feature_maps(original_activations_folder_path,
                          adjusted_activations_folder_path,
                          class_names=None,
                          metrics=None,
                          model=None,
                          train_dataloader=None):
    original_output = get_feature_map_last_layer(original_activations_folder_path)
    adjusted_output = get_feature_map_last_layer(adjusted_activations_folder_path)

    if metrics is None or 'ce' in metrics:
        '''
        We don't want the model's output to change after applying the SAE. The cross-entropy is 
        suitable for comparing probability outputs. Hence, we want the cross-entropy between 
        the original model's output and the output of the modified model to be small.
        '''
        ce = compute_ce(original_output, adjusted_output)
        print(f"Cross-entropy between original model's output and modified model's output: {ce:.4f}")

    if metrics is None or 'percentage_same_classification' in metrics:
        print_percentage_same_classification(original_output, adjusted_output)

    if metrics is None or 'intermediate_feature_maps_similarity' in metrics:
        module_names = get_module_names(model)
        intermediate_feature_maps_similarity(module_names, 
                                             original_activations_folder_path, 
                                             adjusted_activations_folder_path)
        
    if metrics is None or 'train_accuracy' in metrics:
        all_targets = []
        for _, target in train_dataloader:
            all_targets.append(target)
        target = torch.cat(all_targets, dim=0)
        print(f"Shape of target tensor: {target.shape}")
        original_accuracy = get_accuracy(original_output, target)
        adjusted_accuracy = get_accuracy(adjusted_output, target)
        print(f"Train accuracy of original model: {original_accuracy:.4f}")
        print(f"Train accuracy of modified model: {adjusted_accuracy:.4f}")

    if metrics is None or 'visualize_classifications' in metrics:
        show_classification_with_images(train_dataloader, 
                                        class_names,
                                        output=original_output, 
                                        output_2=adjusted_output)