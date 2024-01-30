import os
import torch
import torch.nn.functional as F
import time
import wandb

from utils import load_feature_map, get_classifications, show_classification_with_images, get_module_names, get_stored_number, get_accuracy, get_file_path, log_image_table

def intermediate_feature_maps_similarity(module_names, 
                                         original_activations_folder_path, 
                                         adjusted_activations_folder_path,
                                         model_params,
                                         sae_params,
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
        original_activations_file_path = get_file_path(original_activations_folder_path, layer_name=name, params=model_params, file_name='activations.h5')
        adjusted_activations_file_path = get_file_path(adjusted_activations_folder_path, layer_name=name, params=sae_params, file_name='activations.h5')

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

    for i in range(len(module_names)):
        if cosine_similarity:
            wandb.log({f"cosine_similarity_{module_names[i]}": similarity_mean_list[i]})
            print(f"Layer: {module_names[i]} | Cosine similarity mean: {similarity_mean_list[i]} +/- {similarity_std_list[i]}")
        if L2_distance: 
            wandb.log({f"L2_distance_{module_names[i]}": L2_dist_mean_list[i]})
            print(f"Layer: {module_names[i]} | L2 distance mean of normalized feature maps: {L2_dist_mean_list[i]} +/- {L2_dist_std_list[i]}")
        
def get_feature_map_last_layer(module_names, folder_path, params):
    # the output layer is the last layer
    output_layer = module_names[-1]
    file_path = get_file_path(folder_path, layer_name=output_layer, params=params, file_name='activations.h5')
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
    return F.kl_div(torch.log(input), target, reduction='batchmean')
    # equivalent to: (target * (torch.log(target) - torch.log(input))).sum() / target.size(0)

def evaluate_feature_maps(original_activations_folder_path,
                          adjusted_activations_folder_path,
                          model_params,
                          sae_params,
                          evaluation_results_folder_path=None, # used by show_classification_with_images (if activated)
                          class_names=None,
                          metrics=None,
                          model=None,
                          device=None,
                          train_dataloader=None,
                          layer_name=None):
    module_names = get_module_names(model)
    original_output = get_feature_map_last_layer(module_names, original_activations_folder_path, model_params)
    adjusted_output = get_feature_map_last_layer(module_names, adjusted_activations_folder_path, sae_params)

    print(adjusted_output.shape)
    print(original_output.shape)

    if metrics is None or 'kld' in metrics:
        kld = kl_divergence(adjusted_output, original_output)
        wandb.log({"kld": kld})
        print(f"KL divergence between original model's output and modified model's output: {kld:.4f}")

    if metrics is None or 'percentage_same_classification' in metrics:
        perc = percentage_same_classification(original_output, adjusted_output)
        wandb.log({"percentage_same_classification": perc})
        print(f"Percentage of samples with the same classification (between original and modified model): {perc:.7f}%")

    if metrics is None or 'intermediate_feature_maps_similarity' in metrics:
        intermediate_feature_maps_similarity(module_names, 
                                             original_activations_folder_path, 
                                             adjusted_activations_folder_path,
                                             model_params,
                                             sae_params)
        
    if metrics is None or 'train_accuracy' in metrics:
        batch_file_path = get_file_path(original_activations_folder_path, layer_name=layer_name, params=model_params, file_name='num_batches.txt')
        num_batches = int(get_stored_number(batch_file_path))
        all_targets = []
        batch_idx = 0

        start_time = time.time()
        start_cpu_time = time.process_time()
        for _, target in train_dataloader:
            batch_idx += 1
            all_targets.append(target)
            if batch_idx == num_batches:
                break
        target = torch.cat(all_targets, dim=0)
        print(target.shape)
        original_accuracy = get_accuracy(original_output, target)
        adjusted_accuracy = get_accuracy(adjusted_output, target)
        end_time = time.time()
        end_cpu_time = time.process_time()
        print(f"Time elapsed: {end_time - start_time:.4f} seconds")
        print(f"CPU time elapsed: {end_cpu_time - start_cpu_time:.4f} seconds")
        print(f"Train accuracy of original model: {100*original_accuracy:.4f}%")
        wandb.log({"train_accuracy_original_model": 100*original_accuracy})
        print(f"Train accuracy of modified model: {100*adjusted_accuracy:.4f}%")
        wandb.log({"train_accuracy_modified_model": 100*adjusted_accuracy})

    if metrics is None or 'sparsity' in metrics:
        original_sparsity_file_path = get_file_path(original_activations_folder_path, layer_name=layer_name, params=model_params, file_name='sparsity.txt')
        adjusted_sparsity_file_path = get_file_path(adjusted_activations_folder_path, layer_name=layer_name, params=sae_params, file_name='sparsity.txt')
        original_sparsity = get_stored_number(original_sparsity_file_path)
        print(f"sparsity of the {layer_name} layer output: {100*original_sparsity:.4f}%")
        wandb.log({f"sparsity_layer_{layer_name}_output": 100*original_sparsity})
        adjusted_sparsity = get_stored_number(adjusted_sparsity_file_path)
        print(f"sparsity of the SAE encoder output, i.e., the augmented {layer_name} layer output: {100*adjusted_sparsity:.4f}%")
        wandb.log({f"sparsity_sae_encoder_output_layer_{layer_name}": 100*adjusted_sparsity})
        # sparsity is mean sparsity over all samples (in training data)

    if metrics is None or 'visualize_classifications' in metrics:
        log_image_table(train_dataloader,
                        class_names,
                        output=original_output, 
                        output_2=adjusted_output)
        '''
        show_classification_with_images(train_dataloader, 
                                        class_names,
                                        folder_path=evaluation_results_folder_path,
                                        layer_name=layer_name,
                                        output=original_output, 
                                        output_2=adjusted_output)
        '''