import itertools

# SPECIFY PARAMETERS IN THIS DOCUMENT
# The code will iterate over all possible combinations.
# ALWAYS RUN THIS FILE BEFORE RUNNING main.py!!!

################################################################################################

model_name = ['custom_mlp_1'] #'resnet50'

sae_model_name = ['sae_mlp'] #'sae_conv'

layer_names = [['fc1']] # we have double brackets here so that the list is treated 
# as a single element when creating all possible combinations later on

# run_group_ID we don't specify. On the cluster, it will be passed as input from the .sh script 
# including the cluster process number. Locally, it will be specified in main.py

# if train_sae is set to True, use_sae should also be set to True, but there is also a check for this in the code
use_sae = ['True']

train_sae = ['False']

train_original_model = ['False']

store_activations = ['False'] # store some activations for later analysis as h5 file if desired

compute_feature_similarity = ['True']
# computing feature similarity is likely computationally more expensive than most other tasks
# since it requires computing the similarity between all activations of two models
# so it is recommended to set this to False if you are running a large number of combinations

model_criterion_name = ['cross_entropy']

sae_criterion_name = ['sae_loss']

dead_neurons_epochs = [1]
# measure dead neurons for n epochs, then re-initialize those neurons,
# train for n epochs, then measure dead neurons for the next n epochs etc.

################################################################################################
# DIFFERENT CONFIGURATIONS

# model_epochs and sae_epochs refers to the number of epochs used during training. Even when performing
# inference, one should leave these values as they are, so that the correct trained model (whose file name
# contains the number of epochs with which it was trained) is loaded. By default, when performing inference,
# only 1 epoch is used, see model_pipeline.py. I have not yet implemented a way to perform inference over
# several epochs. Probably it would be best to introduce a new parameter: sae_eval_epochs and model_eval_epochs
# (as opposed to sae_train_epochs and model_train_epochs)

#''' # cifar_10 configuration, cluster, full working example
directory_path = ['/lustre/home/jtoussaint/master_thesis/']
wandb_status = ['True']
model_epochs = [30] 
model_learning_rate = [0.1]
batch_size = [32]
model_optimizer_name = ['sgd']
sae_epochs = [7] 
sae_learning_rate = [0.001]
sae_optimizer_name = ['adam']
sae_batch_size = [64]
sae_lambda_sparse = [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0]
sae_expansion_factor = [2,4,8]
activation_threshold = [0.1]
dataset_name = ['cifar_10']
#'''

''' # cifar_10, cluster, minimal working example
directory_path = ['/lustre/home/jtoussaint/master_thesis/']
wandb_status = ['True']
model_epochs = [3]
model_learning_rate = [0.1, 0.01]
batch_size = [64]
model_optimizer_name = ['sgd']
sae_epochs = [1]
sae_learning_rate = [0.001]
sae_optimizer_name = ['adam']
sae_batch_size = [64]
sae_lambda_sparse = [0.001] 
sae_expansion_factor = [2]
activation_threshold = [0.1]
dataset_name = ['cifar_10'] # cifar_10, mnist
'''

''' 
# cifar_10, local, minimal working example
directory_path = ['C:\\Users\\Jasper\\Downloads\\Master thesis\\Code']
wandb_status = ['False']
model_epochs = [3] # in case of training the model, the first epoch is used for inference to calculate 
# the initial loss (and other values)
model_learning_rate = [0.1]
batch_size = [64]
model_optimizer_name = ['sgd']
sae_epochs = [7] 
sae_learning_rate = [0.001]
sae_optimizer_name = ['adam']
sae_batch_size = [64]
sae_lambda_sparse = [0.001] 
sae_expansion_factor = [2]
activation_threshold = [0.1]
dataset_name = ['cifar_10'] # cifar_10, mnist
'''

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

################################################################################################
# DON'T CHANGE ANYTHING BELOW THIS LINE

# Generate all possible combinations of parameters. However, if we only want to train the 
# original model, we only need to iterate over the parameters relevant for the original model.
# We need to modify this file since the jobs on the cluster are run based on the txt file generated
# by this script, i.e., if we generate 6 combinations, 6 jobs will be run on the cluster.
if train_original_model == ['True']:
    sae_model_name = ['None']
    layer_names = [['None']]
    sae_epochs = ['0']
    sae_learning_rate = ['0']
    sae_optimizer_name = ['None']
    sae_batch_size = ['0']
    sae_lambda_sparse = ['0']
    sae_expansion_factor = ['0']
    compute_feature_similarity = ['False']
    dead_neurons_epochs = ['0']

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
                                    sae_criterion_name,
                                    dead_neurons_epochs)

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
                                            use_sae,
                                            dead_neurons_epochs)

# We write parameters to a separate text file which will be used
# for computing some evaluation metrics.
with open('parameters_eval.txt', 'w') as file_eval:
    for combination in all_combinations_eval:
        line = ','.join(map(str, combination))
        file_eval.write(line + '\n')

print('Parameters written to parameters_eval.txt')