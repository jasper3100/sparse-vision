import torch.nn.functional as F
import torch
import os
import matplotlib.pyplot as plt
import numpy as np

from get_module_names import ModuleNames
from utils import load_model_aux, compute_ce, load_data_aux
from utils_feature_map import load_feature_map

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
                 original_activations_folder_path=None, 
                 adjusted_activations_folder_path=None,
                 weights_folder_path=None):
        self.mode = mode
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.layer_name = layer_name
        self.original_activations_folder_path = original_activations_folder_path
        self.adjusted_activations_folder_path = adjusted_activations_folder_path
        self.weights_folder_path = weights_folder_path
        if self.layer_name is not None:
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
        scores, class_ids = prob.max(dim=1)
        return scores, class_ids

    def get_classification(self, output):

        _, _, self.img_size, self.category_names = load_data_aux(self.dataset_name, 
                                            data_dir=None, 
                                            layer_name=self.layer_name)
        #_, self.weights = load_model_aux(self.model_name, 
        #                                 self.img_size, 
        #                                 expansion_factor=None)
        scores, class_ids = self.classification_results_aux(output)

        #category_list = [self.weights.meta["categories"][index] for index in class_ids]
        self.category_list = [self.category_names[index] for index in class_ids]
        return scores, self.category_list, class_ids
    
    def print_classification(self, output):
        # print final classification of one models
        scores, category_names, _ = self.get_classification(output)
        for i in range(output.size(0)):
            print(f"Sample {i + 1}: {category_names[i]}: {scores[i]:.1f}%")
    
    def show_classification_with_images(self, input_images, target_ids, output):
        _, valloader, img_size, class_names = load_data_aux(self.dataset_name, data_dir=None, layer_name=None)
        scores, predicted_classes, _ = self.get_classification(output)

        number_of_images = 10 # show only the first n images, 
        # for showing all images in the batch use len(predicted_classes)
        fig, axes = plt.subplots(1, number_of_images, figsize=(15, 3))

        for i in range(number_of_images):
            # Unnormalize the image
            img = input_images[i] / 2 + 0.5
            npimg = img.numpy()

            # Display the image with true label, predicted label, and score
            axes[i].imshow(np.transpose(npimg, (1, 2, 0)))
            title = f'True: {class_names[target_ids[i]]}\nPredicted: {predicted_classes[i]} ({scores[i].item():.1f}%)'
            # alternatively: scores.detach().numpy()[i]
            axes[i].set_title(title, fontsize=8)
            axes[i].axis('off')
            
        plt.subplots_adjust(wspace=0.5)  # Adjust space between images
        plt.show()

    def accuracy(self, target, output):
        _, _, class_ids = self.get_classification(output)
        correct_predictions = (class_ids == target).sum().item()
        total_samples = target.size(0)
        accuracy = correct_predictions / total_samples
        print(f'Accuracy: {accuracy * 100:.2f}%')

    def print_classifications(self, original_output, adjusted_output):
        # print final classification of two models (to allow for easier comparison)
        original_score, original_category_names, _ = self.get_classification(original_output)
        adjusted_score, adjusted_category_names, _ = self.get_classification(adjusted_output)
        for i in range(original_output.size(0)):
            print(f"Sample {i + 1}: {original_category_names[i]}: {original_score[i]:.1f}% | {adjusted_category_names[i]}: {adjusted_score[i]:.1f}%")

    def percentage_same_classification(self, original_output, adjusted_output):
        '''
        Calculate percentage of samples with same classification as before
        '''
        _, original_class_ids = self.classification_results_aux(original_output)
        _, adjusted_class_ids = self.classification_results_aux(adjusted_output)
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

    def get_model_output(self, select_data, weights_folder_path):
        _, self.test_data, self.img_size, _ = load_data_aux(self.dataset_name,
                                                            data_dir=None,
                                                            layer_name=None)
        self.model, _ = load_model_aux(self.model_name,
                                       self.img_size, 
                                       expansion_factor=None)

        file_name = 'model_weights.pth'
        weights_file_path = os.path.join(weights_folder_path, file_name)
        self.model.load_state_dict(torch.load(weights_file_path))
        self.model.eval()
            
        if select_data == 'all_eval_data':
            for input, target in self.test_data:
                output = self.model(input)
        
        elif select_data == 'first_batch':
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
                percentage_same_classification = self.percentage_same_classification(original_output, adjusted_output)
                print(f"Percentage of samples with the same classification (between original and modified model): {percentage_same_classification:.1f}%")

            if metrics is None or 'print_classifications' in metrics:
                self.print_classifications(original_output, adjusted_output)

            if metrics is None or 'intermediate_feature_maps_similarity' in metrics:
                self.intermediate_feature_maps_similarity()

        elif self.mode == 'single_model':
            input, target, output = self.get_model_output('first_batch', self.weights_folder_path)
            _, target_all, output_all = self.get_model_output('all_eval_data', self.weights_folder_path)
            #self.print_classification(output)
            self.show_classification_with_images(input, target, output)
            self.accuracy(target_all, output_all)
        

'''
if __name__ == '__main__':
    evaluator = ModelEvaluator(model_name, original_activations_folder_path, adjusted_activations_folder_path, layer_name)
    evaluator.evaluate(metrics=['ce', 'percentage_same_classification', 'print_classifications', 'intermediate_feature_maps_similarity'])
'''