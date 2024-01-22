import torch.nn.functional as F
import torch
import os

from get_module_names import ModuleNames
from utils import load_model_aux, compute_ce, load_data_aux
from utils_feature_map import load_feature_map

class ModifiedModelEvaluator:
    '''
    This class evaluates the modified model (after applying the SAE) and
    compares it to the original model.
    Methods:
        - load feature map of the last layer only
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
        self.module_names = self.get_module_names(self.model_name, self.layer_name)
    
    def get_module_names(self, model_name, layer_name):
        module_names = ModuleNames(model_name)
        return module_names.names_of_main_modules_and_specified_layer(layer_name)

    def load_feature_map_last_layer(self, folder_path):
        # the output layer is the second last layer, because layer_name is the last
        output_layer = self.module_names[-2]
        file_path = os.path.join(folder_path, f'{output_layer}_activations.h5')
        return load_feature_map(file_path)
        
    def classification_results_aux(self, output):
        prob = F.softmax(output, dim=1)
        score, class_ids = prob.max(dim=1)
        return score, class_ids

    def get_classification(self, output):

        _, _, self.img_size = load_data_aux(self.dataset_name, 
                                            data_dir=None, 
                                            layer_name=self.layer_name)
        _, self.weights = load_model_aux(self.model_name, 
                                         self.img_size, 
                                         expansion_factor=None)
        score, class_ids, _ = self.classification_results_aux(output)
        category_names = [self.weights.meta["categories"][index] for index in class_ids]
        return score, category_names
    
    def print_classification(self, output):
        # print final classification of one models
        score, category_names = self.get_classification(output)
        for i in range(output.size(0)):
            print(f"Sample {i + 1}: {category_names[i]}: {score[i]:.1f}%")
    
    def print_classifications(self, original_output, adjusted_output):
        # print final classification of two models (to allow for easier comparison)
        original_score, original_category_names = self.get_classification(original_output)
        adjusted_score, adjusted_category_names = self.get_classification(adjusted_output)
        for i in range(original_output.size(0)):
            print(f"Sample {i + 1}: {original_category_names[i]}: {original_score[i]:.1f}% | {adjusted_category_names[i]}: {adjusted_score[i]:.1f}%")

    def percentage_same_classification(self, original_output, adjusted_output):
        '''
        Calculate percentage of samples with same classification as before
        '''
        _, original_class_ids = self.classification_results_aux(original_output)
        _, adjusted_class_ids = self.classification_results_aux(adjusted_output)
        return (original_class_ids == adjusted_class_ids).sum().item() / original_class_ids.size(0)

    def evaluate(self, metrics=None):
        original_output = self.load_feature_map_last_layer(self.original_activations_folder_path)
        adjusted_output = self.load_feature_map_last_layer(self.adjusted_activations_folder_path)

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