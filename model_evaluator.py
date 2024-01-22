import torch.nn.functional as F
import torch
import os

from get_module_names import ModuleNames
from utils import load_model_aux, compute_ce, load_data_aux
from utils_feature_map import load_feature_map

class ModelEvaluator:
    '''
    Methods:
        - load feature map of original and modified model of the last layer only
        - cross-entropy loss between original model's output and modified model's output
        - print final classification of both models
        - percentage of samples with same classification as before
    '''
    def __init__(self, 
                 model_name, 
                 dataset_name,
                 layer_name,
                 original_activations_folder_path, 
                 adjusted_activations_folder_path):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.layer_name = layer_name
        self.original_activations_folder_path = original_activations_folder_path
        self.adjusted_activations_folder_path = adjusted_activations_folder_path
        self.module_names = self.get_module_names()
    
    def get_module_names(self):
        module_names = ModuleNames(self.model_name)
        return module_names.names_of_main_modules_and_specified_layer(self.layer_name)

    def load_feature_map_last_layer(self):
        # the output layer is the second last layer, because layer_name is the last
        output_layer = self.module_names[-2]

        original_output_file_path = os.path.join(self.original_activations_folder_path, f'{output_layer}_activations.h5')
        adjusted_output_file_path = os.path.join(self.adjusted_activations_folder_path, f'{output_layer}_activations.h5')

        original_output = load_feature_map(original_output_file_path)
        adjusted_output = load_feature_map(adjusted_output_file_path)

        return original_output, adjusted_output

    def classification_results_aux(self, original_output, adjusted_output):
        size = original_output.size(0)
        original_prob = F.softmax(original_output, dim=1)
        adjusted_prob = F.softmax(adjusted_output, dim=1)
        original_score, original_class_ids = original_prob.max(dim=1)
        adjusted_score, adjusted_class_ids = adjusted_prob.max(dim=1)
        return original_score, original_class_ids, adjusted_score, adjusted_class_ids, size

    def print_classifications(self, original_output, adjusted_output):

        _, _, self.img_size = load_data_aux(self.dataset_name, 
                                            data_dir=None, 
                                            layer_name=self.layer_name)
        _, self.weights = load_model_aux(self.model_name, 
                                         self.img_size, 
                                         expansion_factor=None)
        original_score, original_class_ids, adjusted_score, adjusted_class_ids, size = self.classification_results_aux(original_output, adjusted_output)
        original_category_names = [self.weights.meta["categories"][index] for index in original_class_ids]
        adjusted_category_names = [self.weights.meta["categories"][index] for index in adjusted_class_ids]
        for i in range(size):
            print(f"Sample {i + 1}: {original_category_names[i]}: {original_score[i]:.1f}% | {adjusted_category_names[i]}: {adjusted_score[i]:.1f}%")

    def percentage_same_classification(self, original_output, adjusted_output):
        '''
        Calculate percentage of samples with same classification as before
        '''
        _, original_class_ids, _, adjusted_class_ids, _ = self.classification_results_aux(original_output, adjusted_output)
        return (original_class_ids == adjusted_class_ids).sum().item() / original_class_ids.size(0)

    def intermediate_feature_maps_similarity(self):
        '''
        Calculate the similarity between the intermediate feature maps of the original and adjusted model
        We only need to compare the feature maps after the first SAE, because before that they are
        identical.
        '''
        similarity_mean_list = []
        similarity_std_list = []
        L2_dist_mean_list = []
        L2_dist_std_list = []

        for name in self.module_names:
            original_activations_file_path = os.path.join(self.original_activations_folder_path, f'{name}_activations.h5')
            adjusted_activations_file_path = os.path.join(self.adjusted_activations_folder_path, f'{name}_activations.h5')

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

        for i in range(len(self.module_names)):
            print(
                f"Layer: {self.module_names[i]} | Cosine similarity mean: {similarity_mean_list[i]} +/- {similarity_std_list[i]} | L2 distance mean: {L2_dist_mean_list[i]} +/- {L2_dist_std_list[i]}")

    def evaluate(self, metrics=None):
        original_output, adjusted_output = self.load_feature_map_last_layer()

        if metrics is None or 'ce' in metrics:
            '''
            We don't want the model's output to change after applying the SAE. The cross-entropy is 
            suitable for comparing probability outputs. Hence, we want the cross-entropy between 
            the original model's output and the output of the modified model to be small.
            '''
            ce = compute_ce(original_output, adjusted_output)
            print(f"Cross-entropy between original model's output and modified model's output: {ce:.4f}")

        if metrics is None or 'percentage_same_classification' in metrics:
            percentage_same_classification = self.percentage_same_classification(original_output, adjusted_output)
            print(f"Percentage of samples with the same classification (between original and modified model): {percentage_same_classification:.1f}%")

        if metrics is None or 'print_classifications' in metrics:
            self.print_classifications(original_output, adjusted_output)

        if metrics is None or 'intermediate_feature_maps_similarity' in metrics:
            self.intermediate_feature_maps_similarity()

'''
if __name__ == '__main__':
    evaluator = ModelEvaluator(model_name, original_activations_folder_path, adjusted_activations_folder_path, layer_name)
    evaluator.evaluate(metrics=['ce', 'percentage_same_classification', 'print_classifications', 'intermediate_feature_maps_similarity'])
'''