import itertools

# SPECIFY PARAMETERS IN THIS DOCUMENT
# The code will iterate over all possible combinations.
# ALWAYS RUN THIS FILE BEFORE RUNNING main.py!!!

################################################################################################

sae_model_name = ['sae_mlp'] #'sae_conv'

# run_group_ID we don't specify. On the cluster, it will be passed as input from the .sh script 
# including the cluster process number. Locally, it will be specified in main.py

# 0 = False, 1 = True
training = ['1'] # training or inference
original_model = ['0'] # use original model or not
'''
Examples of "sae layers" values
'fc1_fc2__' # original model with SAEs on fc1 and fc2
'__fc3' # take original model and train SAE on fc3
'fc1_fc2__fc3' # take original model with SAEs on fc1 and fc2, and train SAE on fc3
'fc1_fc2__fc3_fc4' # take original model with SAEs on fc1 and fc2, and first train SAE on fc3, then take original model with SAEs on fc1, fc2, fc3 and train SAE on fc4
if we use the original model, sae_layers will be reset to some default value
'''

model_criterion_name = ['cross_entropy']

sae_criterion_name = ['sae_loss']

dead_neurons_steps = [300]#[781] # think about a suitable number... (depends on the dataset)
# f.e. cifar-10, the train dataset has 50,000 samples and with batch size 64, we have 781 batches (where drop_last=True)
# --> 781 training steps
# mnist: 60000 train samples, batch size 64, 937 batches (drop_last=True)
'''
We re-initialize dead neurons, which were dead over the last n steps (n=dead_neurons_steps)
Then, we let the model train with the new neurons for n steps
Then, we measure dead neurons for another n steps and re-initialize the dead neurons from those last n steps etc.
'''
################################################################################################
# DIFFERENT CONFIGURATIONS

# model_epochs and sae_epochs refers to the number of epochs used during training. Even when performing
# inference, one should leave these values as they are, so that the correct trained model (whose file name
# contains the number of epochs with which it was trained) is loaded. By default, when performing inference,
# only 1 epoch is used, see model_pipeline.py. I have not yet implemented a way to perform inference over
# several epochs. Probably it would be best to introduce a new parameter: sae_eval_epochs and model_eval_epochs
# (as opposed to sae_train_epochs and model_train_epochs)

''' # cifar_10 configuration, cluster, full working example
model_name = ['custom_mlp_2']
directory_path = ['/lustre/home/jtoussaint/master_thesis/']
#directory_path = ['C:\\Users\\Jasper\\Downloads\\Master thesis\\Code']
wandb_status = ['1']
#wandb_status = ['0']
model_epochs = [30] 
model_learning_rate = [0.1, 0.01, 0.001]
batch_size = [64, 128, 256, 512, 1024]
model_optimizer_name = ['sgd']
sae_epochs = [20] 
sae_learning_rate = [0.001]
sae_optimizer_name = ['adam']
sae_batch_size = [64]
sae_lambda_sparse = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2] # recall: 1e-3 = 0.001
sae_expansion_factor = [2,4,8]
activation_threshold = [0.1]
dataset_name = ['cifar_10']
'''

''' # cifar_10, cluster, minimal working example
model_name = ['custom_mlp_2']
directory_path = ['/lustre/home/jtoussaint/master_thesis/']
wandb_status = ['1']
model_epochs = [1]
model_learning_rate = [0.1]
batch_size = [64]
model_optimizer_name = ['sgd']
sae_epochs = [2]
sae_learning_rate = [0.001]
sae_optimizer_name = ['adam']
sae_batch_size = [64]
sae_lambda_sparse = [0.001] 
sae_expansion_factor = [2]
activation_threshold = [0.1]
dataset_name = ['cifar_10'] # cifar_10, mnist
'''

#'''# local, MNIST
sae_layers = ['fc1__']
model_name = ['custom_mlp_9']
directory_path = ['C:\\Users\\Jasper\\Downloads\\Master thesis\\Code']
wandb_status = ['0']
model_epochs = [1]
model_learning_rate = [0.1]#,0.2]
batch_size = [64]#,128]
model_optimizer_name = ['sgd']
sae_epochs = [1] 
sae_learning_rate = [0.001]
sae_optimizer_name = ['adam']
sae_batch_size = [64]
sae_lambda_sparse = [1e-1] #,1e-2] 
sae_expansion_factor = [2]#,4]
activation_threshold = [0.1]
dataset_name = ['mnist'] # cifar_10, mnist
#'''

''' cluster, MNIST
sae_layers = ['fc1__']
model_name = ['custom_mlp_10']#custom_mlp_8', 'custom_mlp_9', 'custom_mlp_10']
directory_path = ['/lustre/home/jtoussaint/master_thesis/']
wandb_status = ['1']
model_epochs = [10]
model_learning_rate = [0.1] #[0.0001] #[0.001,0.0001] # 0.1 for MLP
batch_size = [64] #[64,128] # 64 for MLP
model_optimizer_name = ['sgd'] #sgd for MLP
sae_epochs = [15]
sae_learning_rate = [0.001]
sae_optimizer_name = ['adam']
sae_batch_size = [64]
sae_lambda_sparse = [5,10,20,40,80]#[0.5,1,2,3,4,5,7,10]#[1e-4, 1e-3, 1e-2, 1e-1] # recall: 1e-3 = 0.001
sae_expansion_factor = [4,8,16,32]
activation_threshold = [0.1]
dataset_name = ['mnist'] # cifar_10, mnist
'''

'''
# local, Tiny ImageNet
sae_layers = ['layer1.0.conv1__'] 
model_name = ['resnet18']
directory_path = ['C:\\Users\\Jasper\\Downloads\\Master thesis\\Code']
wandb_status = ['0']
model_epochs = [2] #7
model_learning_rate = [0.001]
batch_size = [100]
model_optimizer_name = ['sgd_w_scheduler']
sae_epochs = [1] 
sae_learning_rate = [0.001]
sae_optimizer_name = ['adam']
sae_batch_size = [100]
sae_lambda_sparse = [1e-1] 
sae_expansion_factor = [2]
activation_threshold = [0.1]
dataset_name = ['tiny_imagenet'] # cifar_10, mnist
'''

#''' # cluster, Tiny ImageNet
#sae_layers = ['layer1.0.conv1__'] 
sae_layers = ['__layer3.0.conv2']
model_name = ['resnet18']
directory_path = ['/lustre/home/jtoussaint/master_thesis/']
wandb_status = ['1']
model_epochs = [10]
model_learning_rate = [0.001]
batch_size = [100]
model_optimizer_name = ['sgd_w_scheduler']
sae_epochs = [5] 
sae_learning_rate = [0.001]#, 0.0001]
sae_optimizer_name = ['adam']
sae_batch_size = [64]
sae_lambda_sparse = [0.1,0.5,2,5] # 0.1
sae_expansion_factor = [2,4,8] # 8
activation_threshold = [0.001]
dataset_name = ['tiny_imagenet'] # cifar_10, mnist
#'''

################################################################################################
# DON'T CHANGE ANYTHING BELOW THIS LINE

training = ['True'] if training == ['1'] else ['False']
original_model = ['True'] if original_model == ['1'] else ['False']

# Generate all possible combinations of parameters. However, if we only want to train or evaluate the 
# original model, we only need to iterate over the parameters relevant for the original model.
# We need to modify this file since the jobs on the cluster are run based on the txt file generated
# by this script, i.e., if we generate 6 combinations, 6 jobs will be run on the cluster.
if original_model == ['True']:
    sae_model_name = ['None']
    sae_layers = ['None__']
    sae_epochs = ['0']
    sae_learning_rate = ['0']
    sae_optimizer_name = ['None']
    sae_batch_size = ['0']
    sae_lambda_sparse = ['0']
    sae_expansion_factor = ['1'] # we computing the sparsity we divide by this number, so it should be 1 if we don't use SAE

all_combinations = itertools.product(model_name,
                                    sae_model_name,
                                    sae_layers,
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
                                    training,
                                    original_model,
                                    model_criterion_name,
                                    sae_criterion_name,
                                    dead_neurons_steps)
# Write the combinations to a text file which will be refered to
# by the .sh script file (if running the job on the cluster) and 
# the main.py file (if running the job locally)
number_combinations = 0
with open('parameters.txt', 'w') as file:
    for combination in all_combinations:
        line = ','.join(map(str, combination))
        file.write(line + '\n')
        number_combinations += 1

print(f'{number_combinations} parameter combinations written to parameters.txt')

# Now we create another such txt file for evaluating some results.
all_combinations_eval = itertools.product(model_name,
                                          sae_model_name,
                                            sae_layers, 
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
                                            activation_threshold,
                                            dataset_name,
                                            original_model, 
                                            dead_neurons_steps)
# We write parameters to a separate text file which will be used
# for computing some evaluation metrics.
number_combinations = 0
with open('parameters_eval.txt', 'w') as file_eval:
    for combination in all_combinations_eval:
        line = ','.join(map(str, combination))
        file_eval.write(line + '\n')
        number_combinations += 1

print(f'{number_combinations} parameter combinations written to parameters_eval.txt')