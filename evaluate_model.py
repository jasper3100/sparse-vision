import torch.nn.functional as F
import torch
import os

from utils import load_model_aux, compute_ce, load_data_aux
from utils_names import load_module_names
from utils_feature_map import load_feature_map
from evaluation.feature_map_similarity import intermediate_feature_maps_similarity
from evaluation.show_classification_images import show_classification_with_images

class ModelEvaluator:
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
                 mode,
                 model_name, 
                 dataset_name,
                 layer_name=None,
                 batch_size=32,
                 original_activations_folder_path=None, 
                 adjusted_activations_folder_path=None,
                 weights_folder_path=None):
        self.mode = mode
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.layer_name = layer_name
        self.batch_size = batch_size
        self.original_activations_folder_path = original_activations_folder_path
        self.adjusted_activations_folder_path = adjusted_activations_folder_path
        self.weights_folder_path = weights_folder_path
        if self.layer_name is not None:
            self.module_names = load_module_names(self.model_name, self.dataset_name, self.layer_name)
        self.train_data, self.test_data, self.img_size, self.class_names = load_data_aux(self.dataset_name, 
                                                                                        self.batch_size,
                                                                                        data_dir=None, 
                                                                                        layer_name=self.layer_name)

    def load_feature_map_last_layer(self, folder_path):
        # the output layer is the last layer
        output_layer = self.module_names[-1]
        file_path = os.path.join(folder_path, f'{output_layer}_activations.h5')
        return load_feature_map(file_path)
    
    def classification_results_aux(self, output):
        prob = F.softmax(output, dim=1)
        scores, class_ids = prob.max(dim=1)
        return scores, class_ids

    def get_classification(self, output):
        #_, self.weights = load_model_aux(self.model_name, 
        #                                 self.img_size, 
        #                                 expansion_factor=None)
        scores, class_ids = self.classification_results_aux(output)

        #category_list = [self.weights.meta["categories"][index] for index in class_ids]
        self.category_list = [self.class_names[index] for index in class_ids]
        return scores, self.category_list, class_ids
    
    def print_classification(self, output):
        # print final classification of one models
        scores, category_names, _ = self.get_classification(output)
        for i in range(output.size(0)):
            print(f"Sample {i + 1}: {category_names[i]}: {scores[i]:.1f}%")

    def accuracy(self):
        for input, target in self.test_data:
            output = self.model(input)
        _, _, class_ids = self.get_classification(output)
        correct_predictions = (class_ids == target).sum().item()
        total_samples = target.size(0)
        accuracy = correct_predictions / total_samples
        print(f'Accuracy: {accuracy * 100:.2f}%')

    def show_classification_with_images_aux(self, input_images, target_ids, output, output_2=None):
        scores, predicted_classes, _ = self.get_classification(output)
        print(output.shape)
        if output_2 is not None:
            scores_2, predicted_classes_2, _ = self.get_classification(output_2)
            show_classification_with_images(input_images, target_ids, self.class_names, scores, predicted_classes, scores_2, predicted_classes_2)
        else:
            show_classification_with_images(input_images, target_ids, self.class_names, scores, predicted_classes)

    def print_classifications(self, original_output, adjusted_output):
        '''
        Print final classification of two models (to allow for easier comparison)
        '''
        original_score, original_category_names, _ = self.get_classification(original_output)
        adjusted_score, adjusted_category_names, _ = self.get_classification(adjusted_output)
        for i in range(10): #original_output.size(0)):
            print(f"Sample {i + 1}: {original_category_names[i]}: {original_score[i]:.1f}% | {adjusted_category_names[i]}: {adjusted_score[i]:.1f}%")

    def percentage_same_classification(self, original_output, adjusted_output):
        '''
        Calculate percentage of samples with same classification as before
        '''
        _, original_class_ids = self.classification_results_aux(original_output)
        _, adjusted_class_ids = self.classification_results_aux(adjusted_output)
        percentage_same_classification = (original_class_ids == adjusted_class_ids).sum().item() / original_class_ids.size(0)
        print(f"Percentage of samples with the same classification (between original and modified model): {percentage_same_classification:.7f}%")

    def get_model_output_first_batch(self, weights_folder_path):
        # for the first batch of the TEST DATA: get input_images, target, and predicted output
        self.model, _ = load_model_aux(self.model_name,
                                       self.img_size, 
                                       expansion_factor=None)
        file_name = 'model_weights.pth'
        weights_file_path = os.path.join(weights_folder_path, file_name)
        self.model.load_state_dict(torch.load(weights_file_path))
        self.model.eval()
        input, target = next(iter(self.test_data))
        output = self.model(input)
        return input, target, output

    def evaluate(self, metrics=None):
        if self.mode == 'compare_models':
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
                self.percentage_same_classification(original_output, adjusted_output)

            if metrics is None or 'print_classifications' in metrics:
                self.print_classifications(original_output, adjusted_output)

            if metrics is None or 'intermediate_feature_maps_similarity' in metrics:
                intermediate_feature_maps_similarity(self.module_names, self.original_activations_folder_path, self.adjusted_activations_folder_path)

            if metrics is None or 'visualize_classifications' in metrics:
                input_images, targets = next(iter(self.train_data))                
                print(input_images.shape)
                print(targets.shape)
                self.show_classification_with_images_aux(input_images, targets, original_output, adjusted_output)

        elif self.mode == 'single_model':
            input, target, output = self.get_model_output_first_batch(self.weights_folder_path)
            #self.print_classification(output)
            self.show_classification_with_images_aux(input, target, output)
            self.accuracy()
        

'''
if __name__ == '__main__':
    evaluator = ModelEvaluator(model_name, original_activations_folder_path, adjusted_activations_folder_path, layer_name)
    evaluator.evaluate(metrics=['ce', 'percentage_same_classification', 'print_classifications', 'intermediate_feature_maps_similarity'])
'''