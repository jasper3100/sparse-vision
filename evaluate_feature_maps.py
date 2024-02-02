import os
import torch
import torch.nn.functional as F
import time
import wandb

#from utils import load_feature_map, get_classifications, show_classification_with_images, get_module_names, get_stored_numbers, calculate_accuracy, get_file_path, log_image_table, compute_sparsity, get_target_output
from utils import *

def intermediate_feature_maps_similarity(module_names, 
                                         original_activations_folder_path, 
                                         adjusted_activations_folder_path,
                                         model_params,
                                         sae_params,
                                         wandb_status,
                                         L2_distance=True,
                                         cosine_similarity=False):
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
        original_activations_file_path = get_file_path(original_activations_folder_path, layer_names=[name], params=model_params, file_name='activations.h5')
        adjusted_activations_file_path = get_file_path(adjusted_activations_folder_path, layer_names=[name], params=sae_params, file_name='activations.h5')

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

        if L2_distance:
            # calculate euclidean distance between feature maps
            # first, we normalize the feature maps, because otherwise the results are not comparable
            # between feature maps with large values and small values (f.e. softmax layer output has small values)
            normalized_original_feature_map = F.normalize(original_feature_map)
            normalized_adjusted_feature_map = F.normalize(adjusted_feature_map)
            L2_dist = torch.linalg.norm(normalized_original_feature_map - normalized_adjusted_feature_map, dim=1)
            L2_dist_mean = round(L2_dist.mean().item(), 2)
            L2_dist_std = round(L2_dist.std().item(), 2)        
            L2_dist_mean_list.append(L2_dist_mean)
            L2_dist_std_list.append(L2_dist_std)

    if wandb_status:
        table_similarity = wandb.Table(columns=["Layer", "L2 distance mean", "L2 distance std"], data=[[module_names[i], L2_dist_mean_list[i], L2_dist_std_list[i]] for i in range(len(module_names))])
        wandb.log({"L2 distance": table_similarity}, commit=False)

    for i in range(len(module_names)):
        if cosine_similarity:
            #wandb.log({f"cosine_similarity_{module_names[i]}": similarity_mean_list[i]})
            print(f"Layer: {module_names[i]} | Cosine similarity mean: {similarity_mean_list[i]} +/- {similarity_std_list[i]}")
        if L2_distance: 
            #wandb.log({f"L2_distance_{module_names[i]}": L2_dist_mean_list[i]})
            print(f"Layer: {module_names[i]} | L2 distance mean of normalized feature maps: {L2_dist_mean_list[i]} +/- {L2_dist_std_list[i]}")
        
def get_feature_map_last_layer(module_names, folder_path, params):
    # the output layer is the last layer
    output_layer = module_names[-1]
    file_path = get_file_path(folder_path, layer_names=[output_layer], params=params, file_name='activations.h5')
    return load_feature_map(file_path)

def percentage_same_classification(original_output, adjusted_output):
    '''
    Print percentage of samples with same classification as before
    '''
    _, _, original_class_ids = get_classifications(original_output)
    _, _, adjusted_class_ids = get_classifications(adjusted_output)
    return 100 * ((original_class_ids == adjusted_class_ids).sum().item() / original_class_ids.size(0))

def kl_divergence(input, target):
    '''
    Compute the KL divergence between two probability distributions.
    '''
    # we add a small constant to avoid having log(0)
    return F.kl_div(torch.log(input+1e-8), target+1e-8, reduction='batchmean')
    # equivalent to: (target * (torch.log(target) - torch.log(input))).sum() / target.size(0)


def polysemanticity_level(encoder_output, target, class_names, activation_threshold):
    # targets is of the form [6,9,9,1,2,...], i.e., it contains the target class indices
    # Let c denote the number of classes.
    c = len(class_names)
    # target shape [number of samples n]
    #n = target.shape[0]
    # encoder_output.shape [number of samples n, number of neurons in augmented layer d]
    d = encoder_output.shape[1]

    # Create a binary mask for values above the activation_threshold
    above_threshold = encoder_output > activation_threshold
    # We create a matrix of size [d,c] where each row i, contains for a certain
    # dimension i of all activations, the number of times a class j has an activation
    # above the threshold.
    counting_matrix = torch.zeros(d, c)
    above_threshold = above_threshold.to(counting_matrix.dtype)  # Convert to the same type
    counting_matrix.index_add_(1, target, above_threshold.t())
    # The code is equivalent to the below for loop (which is too slow though)
    '''
    for i in range(n):
        for j in range(d):
            counting_matrix[j, target[i]] += encoder_output[i, j] > activation_threshold
    '''

    # Now, for each row i (each dimension i of the activations), we count the number of distinct positive integers, i.e.,
    # the number of classes that have an activation above the threshold
    # .bool() turns every non-zero element into a 1, and every zero into a 0
    distinct_counts = torch.sum(counting_matrix.bool(), dim=1)

    # We calculate the mean and standard deviation of the number of classes, that one neuron is active on
    mean_distinct_counts = distinct_counts.float().mean().item()
    std_distinct_counts = distinct_counts.float().std().item()

    return mean_distinct_counts, std_distinct_counts

def evaluate_feature_maps(original_activations_folder_path,
                          adjusted_activations_folder_path,
                          model_params,
                          sae_params,
                          wandb_status,
                          evaluation_results_folder_path=None, # used by show_classification_with_images (if activated)
                          class_names=None,
                          metrics=None,
                          model=None,
                          device=None,
                          train_dataloader=None,
                          layer_names=None,
                          train_dataset_length=None,
                          encoder_output_folder_path=None,
                          activation_threshold=None):
    if len(layer_names) > 1:
        raise ValueError("So far, only one layer can be specified for evaluation")
        # TO-DO!!!

    module_names = get_module_names(model)
    original_output = get_feature_map_last_layer(module_names, original_activations_folder_path, model_params)
    adjusted_output = get_feature_map_last_layer(module_names, adjusted_activations_folder_path, sae_params)

    original_output = F.softmax(original_output, dim=1)
    adjusted_output = F.softmax(adjusted_output, dim=1)

    print(adjusted_output.shape)
    print(original_output.shape)

    target = get_target_output(device, 
                                train_dataloader, 
                                original_activations_folder_path=original_activations_folder_path,
                                layer_names=layer_names,
                                model_params=model_params)
    #print(target.shape)

    if metrics is None or 'kld' in metrics:
        kld = kl_divergence(adjusted_output, original_output)
        if wandb_status:
            wandb.log({"kld": kld})
        print(f"KL divergence between original model's output and modified model's output: {kld:.4f}")
        pass
    
    if metrics is None or 'percentage_same_classification' in metrics:
        perc = percentage_same_classification(original_output, adjusted_output)
        if wandb_status:
            wandb.log({"percentage_same_classification": perc})
        print(f"Percentage of samples with the same classification (between original and modified model): {perc:.7f}%")

    if metrics is None or 'intermediate_feature_maps_similarity' in metrics:
        intermediate_feature_maps_similarity(module_names, 
                                             original_activations_folder_path, 
                                             adjusted_activations_folder_path,
                                             model_params,
                                             sae_params,
                                             wandb_status)
        
    if metrics is None or 'train_accuracy' in metrics:
        original_accuracy = calculate_accuracy(original_output, target)
        adjusted_accuracy = calculate_accuracy(adjusted_output, target)
        #end_time = time.time()
        #end_cpu_time = time.process_time()
        #print(f"Time elapsed: {end_time - start_time:.4f} seconds")
        #print(f"CPU time elapsed: {end_cpu_time - start_cpu_time:.4f} seconds")
        print(f"Train accuracy of original model: {100*original_accuracy:.4f}%")
        #wandb.log({"train_accuracy_original_model": 100*original_accuracy})
        print(f"Train accuracy of modified model: {100*adjusted_accuracy:.4f}%")
        #wandb.log({"train_accuracy_modified_model": 100*adjusted_accuracy})
        if wandb_status:
            table_accuracy = wandb.Table(columns=["Model", "Train accuracy"], data=[["Original model", 100*original_accuracy], ["Modified model", 100*adjusted_accuracy]])
            wandb.log({"train_accuracy": table_accuracy}, commit=False)

    if metrics is None or 'sparsity' in metrics:
        original_sparsity_file_path = get_file_path(original_activations_folder_path, layer_names=layer_names, params=model_params, file_name='sparsity.txt')
        adjusted_sparsity_file_path = get_file_path(adjusted_activations_folder_path, layer_names=layer_names, params=sae_params, file_name='sparsity.txt')

        #print(original_sparsity_file_path)
        #print(get_stored_numbers(original_sparsity_file_path))

        original_activated_units, original_total_units = get_stored_numbers(original_sparsity_file_path)
        original_sparsity = compute_sparsity(original_activated_units, original_total_units)
        mean_original_activated_units = int(original_activated_units/train_dataset_length)
        mean_original_total_units = int(original_total_units/train_dataset_length)
        print(f"Layer {layer_names}: on average, {mean_original_activated_units} out of {mean_original_total_units} units are activated")
        print(f"Sparsity of the {layer_names} layer output: {100*original_sparsity:.4f}%")

        adjusted_activated_units, adjusted_total_units = get_stored_numbers(adjusted_sparsity_file_path)
        adjusted_sparsity = compute_sparsity(adjusted_activated_units, adjusted_total_units, expansion_factor=sae_params['expansion_factor'])
        mean_adjusted_activated_units = int(adjusted_activated_units/train_dataset_length)
        mean_adjusted_total_units = int(adjusted_total_units/train_dataset_length)
        # equivalently we could do: compute_sparsity(adjusted_activated_units, original_total_units)
        print(f"Augmented layer {layer_names}: on average, {mean_adjusted_activated_units} out of {mean_adjusted_total_units} units are activated")
        print(f"Sparsity of the SAE encoder output, i.e., the augmented {layer_names} layer output, relative to the size of layer {layer_names}: {100*adjusted_sparsity:.4f}%")

        # sparsity is mean sparsity over all samples (in training data)
        if wandb_status:
            table_sparsity = wandb.Table(columns=["Layer", "Average number of activating units in layer/Number of total units in layer", "Sparsity relative to size of original layer"], 
                                         data=[[layer_names, f"{mean_original_activated_units}/{mean_original_total_units}", 100*original_sparsity], 
                                               [f"SAE encoder output layer {layer_names}", f"{mean_adjusted_activated_units}/{mean_adjusted_total_units}", 100*adjusted_sparsity]])
            wandb.log({"sparsity": table_sparsity}, commit=False)

    if metrics is None or 'polysemanticity_level' in metrics:
        encoder_output_file_path = get_file_path(encoder_output_folder_path, layer_names=layer_names, params=sae_params, file_name='activations.h5')
        encoder_output = load_feature_map(encoder_output_file_path)

        mean_active_classes_per_neuron, std_active_classes_per_neuron = polysemanticity_level(encoder_output, target, class_names, activation_threshold)
        print(f"Mean number of active classes per neuron in the augmented layer: {mean_active_classes_per_neuron:.4f}")
        print(f"Standard deviation of the number of active classes per neuron in the augmented layer: {std_active_classes_per_neuron:.4f}")
        if wandb_status:
            wandb.log({"mean_active_classes_per_neuron": mean_active_classes_per_neuron, "std_active_classes_per_neuron": std_active_classes_per_neuron})        

    if metrics is None or 'visualize_classifications' in metrics:
        if wandb_status:
            log_image_table(train_dataloader,
                            class_names,
                            output=original_output, 
                            output_2=adjusted_output)
        else:
            show_classification_with_images(train_dataloader, 
                                            class_names,
                                            folder_path=evaluation_results_folder_path,
                                            layer_names=layer_names,
                                            output=original_output, 
                                            output_2=adjusted_output)