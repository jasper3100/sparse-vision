import os

'''
Contents of the repository: 

- main.py: main script to define parameters and run the pipeline

- resnet50.py: instantiate pre-trained ResNet50 model
- data.py: dataset
- sae.py: sparse autoencoder model
- sparse_loss.py: custom loss function for the sparse autoencoder
- aux.py: auxiliary functions: print classification results of model and print all layer names

- extract_intermediate_features.py: extract intermediate features of a specific layer of the model
- train_sae.py: train sparse autoencoder 
- evaluate_model_on_adjusted_features.py: output of intermediate layer --> autoencoder --> into model
'''

# TO-DO: MAKE THE PARAMETERS PROPER PARAMETERS; DO NOT IMPORT THEM TO OTHER SCRIPTS THROUGH "IMPORT"!!!

model_name = 'resnet50' # for seeing all possible models, see model.py
dataset_name = 'sample_data_1' # for seeing all possible datasets, see data.py
layer_name = 'model.layer1[0].conv3' # for seeing all possible layers, run the get_names_of_all_layers() function in aux.py
expansion_factor = 2 # specifies by how much the number of channels in the SAE should be expanded

directory_path = r'C:\Users\Jasper\Downloads\Master thesis\Code'

# Do not edit below this line
activations_folder_path = os.path.join(directory_path, 'intermediate_feature_maps', model_name, dataset_name)
activations_file_path = os.path.join(activations_folder_path, f'{layer_name}_intermediate_activations.json')

sae_weights_folder_path = os.path.join(directory_path, 'trained_sae_weights', model_name, dataset_name)
sae_weights_file_path = os.path.join(sae_weights_folder_path, f'{layer_name}_trained_sae_weights.pth')