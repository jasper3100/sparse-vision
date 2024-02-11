import itertools

# Specify parameters here
# The code will iterate over all possible combinations
model_name = ['custom_mlp_1'] #'resnet50'
sae_model_name = ['sae_mlp'] #'sae_conv'
layer_names = [['fc1']] # we have double brackets here so that the list is treated 
# as a single element when creating all possible combinations later on
# run_group_ID we don't specify. On the cluster, it will be passed as input from the .sh script 
# including the cluster process number. Locally, it will be specified in main.py

''' # cifar_10 configuration, cluster
directory_path = ['/lustre/home/jtoussaint/master_thesis/']
wandb_status = ['True']
model_epochs = [30]
model_learning_rate = [0.1]
batch_size = [32]
model_optimizer_name = ['sgd']
sae_epochs = [15]
sae_learning_rate = [0.001]
sae_optimizer_name = ['adam']
sae_batch_size = [64]
sae_lambda_sparse = [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0]
sae_expansion_factor = [2,4,8]
activation_threshold = [0.1]
dataset_name = ['cifar_10']
'''

#''' # cifar_10, local
directory_path = [r'C:\Users\Jasper\Downloads\Master thesis\Code']
wandb_status = ['False']
model_epochs = [1]
model_learning_rate = [0.1]
batch_size = [32]
model_optimizer_name = ['sgd']
sae_epochs = [1] 
sae_learning_rate = [0.001]
sae_optimizer_name = ['adam']
sae_batch_size = [64]
sae_lambda_sparse = [0.001] 
sae_expansion_factor = [2]
activation_threshold = [0.1]
dataset_name = ['cifar_10'] # cifar_10, mnist
#'''

''' # last configuration for MNIST that I used, cluster
directory_path = ['/lustre/home/jtoussaint/master_thesis/']
wandb_status = ['True']
model_epochs = [5] #[1, 2]
model_learning_rate = [0.1] #[0.1, 0.2]
batch_size = [32]
model_optimizer_name = ['sgd']
sae_epochs = [5] #[1, 2]
sae_learning_rate = [0.001,0.1]
sae_optimizer_name = ['adam']
sae_batch_size = [64]
sae_lambda_sparse = [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0]
sae_expansion_factor = [2,4,8]
activation_threshold = [0.1]
dataset_name = ['mnist'] # cifar_10, mnist
'''

use_sae = ['True']
train_sae = ['False']
train_original_model = ['False']
store_activations = ['False']
compute_feature_similarity = ['True']
# computing feature similarity is likely computationally more expensive than most other tasks
model_criterion_name = ['cross_entropy']
sae_criterion_name = ['sae_loss']

# Generate all possible combinations
all_combinations = itertools.product(model_name,
                                    sae_model_name,
                                    layer_names,
                                    directory_path,
                                    wandb_status,
                                    model_epochs,
                                    model_learning_rate,
                                    batch_size,
                                    model_optimizer_name,
                                    sae_epochs,
                                    sae_learning_rate,
                                    sae_optimizer_name,
                                    sae_batch_size,
                                    sae_lambda_sparse,
                                    sae_expansion_factor,
                                    activation_threshold,
                                    dataset_name,
                                    use_sae,
                                    train_sae,
                                    train_original_model,
                                    store_activations,
                                    compute_feature_similarity,
                                    model_criterion_name,
                                    sae_criterion_name)

# Write the combinations to a text file which will be refered to
# by the .sh script file (if running the job on the cluster) and 
# the main.py file (if running the job locally)
with open('parameters.txt', 'w') as file:
    for combination in all_combinations:
        line = ','.join(map(str, combination))
        file.write(line + '\n')

print('Parameters written to parameters.txt')

# Now we create another such txt file for evaluating some results.
all_combinations_eval = itertools.product(model_name,
                                          sae_model_name,
                                            layer_names,
                                            directory_path,
                                            wandb_status,
                                            sae_epochs,
                                            sae_learning_rate,
                                            sae_optimizer_name,
                                            sae_batch_size,
                                            activation_threshold,
                                            dataset_name,
                                            use_sae)

# We write parameters to a separate text file which will be used
# for computing some evaluation metrics.
with open('parameters_eval.txt', 'w') as file_eval:
    for combination in all_combinations_eval:
        line = ','.join(map(str, combination))
        file_eval.write(line + '\n')

print('Parameters written to parameters_eval.txt')