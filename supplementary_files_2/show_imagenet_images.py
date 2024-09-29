from utils import *

# ALWAYS CHANGE THE DIRECTORY PATH TO LOCAL OR CLUSTER

#directory_path = 'C:\\Users\\Jasper\\Downloads\\Master thesis\\code'
directory_path = '/lustre/home/jtoussaint/master_thesis/'





model_name = 'inceptionv1'
dataset_name = 'imagenet'
sae_model_name = 'sae_mlp'
model_weights_folder_path, sae_weights_folder_path, evaluation_results_folder_path = get_folder_paths(directory_path, model_name, dataset_name, sae_model_name)

model_key = "sae"
layer_name = "mixed3a"
epoch = '6'

# just for getting the right file name
model_epochs = '1'
model_learning_rate = '0.001'
batch_size = '512'
model_optimizer_name = 'sgd'

sae_epochs = '10'
sae_learning_rate = '0.001'
sae_optimizer_name = 'constrained_adam'
sae_batch_size = '256'
sae_lambda_sparse = '5.0'
sae_expansion_factor = '8'

dead_neurons_steps = '199'

model_params = {'model_name': model_name, 'epochs': model_epochs, 'learning_rate': model_learning_rate, 'batch_size': batch_size, 'optimizer': model_optimizer_name}
sae_params = {'sae_model_name': sae_model_name, 'sae_epochs': sae_epochs, 'learning_rate': sae_learning_rate, 'batch_size': sae_batch_size, 'optimizer': sae_optimizer_name, 'expansion_factor': sae_expansion_factor, 
                           'lambda_sparse': sae_lambda_sparse, 'dead_neurons_steps': dead_neurons_steps}

model_params_temp = {k: str(v) for k, v in model_params.items()}
sae_params_temp = {k: str(v) for k, v in sae_params.items()}
params_string = '_'.join(model_params_temp.values()) + "_" + "_".join(sae_params_temp.values())

device = torch.device('cpu')

unit_idx = 1

folder_path = os.path.join(evaluation_results_folder_path, 'filename_indices')
file_path = get_file_path(folder_path=folder_path,
                        sae_layer=model_key + '_' + layer_name,
                        params=params_string,
                        file_name=f'max_min_filename_indices_epoch_{epoch}.pt')
data = torch.load(file_path, map_location=device)
max_filename_indices = data['max_filename_indices']
min_filename_indices = data['min_filename_indices']

indices = max_filename_indices[:,unit_idx]

print(indices)
print(indices.shape)

# print max value in max_filename_indices
max_value = torch.max(indices)
print(max_value)

# print min value in min_filename_indices
min_value = torch.min(indices)
print(min_value)
'''
suggests that indices were taken over a batch of 512 images
model batch size is 512 (since i dont use sae right now), this is the batch size I'm using
--> fix: filename indices need to be not relative to a batch but for all files in general....
# needs a throrough fix...
'''

show_imagenet_images(unit_idx, directory_path, params_string, 
                     epoch=epoch,
                     evaluation_results_folder_path=evaluation_results_folder_path, 
                     model_key=model_key, 
                     layer_name=layer_name, 
                     device=device)