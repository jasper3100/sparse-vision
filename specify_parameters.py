import itertools, functools, operator

# SPECIFY PARAMETERS IN THIS DOCUMENT
# The code will iterate over all possible combinations.
# ALWAYS RUN THIS FILE BEFORE RUNNING main.py!!!

################################################################################################

sae_model_name = ['sae_mlp'] #['sae_mlp'] #'sae_conv' # gated_sae

# run_group_ID we don't specify. On the cluster, it will be passed as input from the .sh script 
# including the cluster process number. Locally, it will be specified in main.py

# 0 = False, 1 = True
training = ['0'] # training or inference
original_model = ['0'] # use original model or not
mis = ['0'] # 0: no, 1: store values for MIS, 2: compute MIS
#compute_ie = ['40'] # 0: no, 1: store average values, 2: compute IE of nodes, 3: compute IE of edges, 4: compute faithfulness
# generate a list of numbers from 0 to 25, turn each number into string and add a leading 4 to each string
compute_ie = [str(4) + str(i) for i in range(20)]

model_criterion_name = ['cross_entropy'] #['negative_log_likelihood'] #['cross_entropy']

sae_criterion_name = ['sae_loss'] #['sae_loss'] # gated_sae_loss

dead_neurons_steps = [194] # [625] #, 9912]#[781] # think about a suitable number... (depends on the dataset)
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
dataset_name = ['cifar_10'] # cifar_10, mnist
'''

#'''# local, MNIST
sae_layer = ['fc1']
model_name = ['custom_mlp_9']
directory_path = ['C:\\Users\\Jasper\\Downloads\\Master thesis\\Code']
wandb_status = ['0']
model_epochs = [1]
model_learning_rate = [0.1]#,0.2]
batch_size = [64]#,128]
model_optimizer_name = ['sgd']

sae_epochs = [2] 
sae_learning_rate = [0.001]
sae_optimizer_name = ['adam']
sae_batch_size = [64]
sae_lambda_sparse = [1e-1] #,1e-2] 
sae_expansion_factor = [4]#[2,4,6]#,4]
dataset_name = ['mnist'] # cifar_10, mnist
sae_checkpoint_epoch = [1] # 0 means no checkpoint
#'''

''' cluster, MNIST
sae_layer = ['fc1']
model_name = ['custom_mlp_10']#custom_mlp_8', 'custom_mlp_9', 'custom_mlp_10']
directory_path = ['/lustre/home/jtoussaint/master_thesis/']
wandb_status = ['1']
model_epochs = [10]
model_learning_rate = [0.1] #[0.0001] #[0.001,0.0001] # 0.1 for MLP
batch_size = [64] #[64,128] # 64 for MLP
model_optimizer_name = ['sgd'] #sgd for MLP

sae_epochs = [2]
sae_learning_rate = [0.001]
sae_optimizer_name = ['adam']
sae_batch_size = [64]
sae_lambda_sparse = [11]#[1e-4, 1e-3, 1e-2, 1e-1] # recall: 1e-3 = 0.001
sae_expansion_factor = [2] #[4,8,16,32]
dataset_name = ['mnist'] # cifar_10, mnist
sae_checkpoint_epoch = [1] # 0 means no checkpoint
'''

'''
# local, Tiny ImageNet
sae_layer = ['layer1.0.conv1'] 
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
dataset_name = ['tiny_imagenet'] # cifar_10, mnist
'''

''' # cluster, Tiny ImageNet
#sae_layer = ['layer1.0.conv1'] 
sae_layer = ['layer3.0.conv2']
model_name = ['resnet18']
directory_path = ['/lustre/home/jtoussaint/master_thesis/']
wandb_status = ['1']
model_epochs = [2]
model_learning_rate = [0.001]
batch_size = [100]
model_optimizer_name = ['sgd_w_scheduler']
sae_epochs = [7] 
sae_learning_rate = [0.001]#, 0.0001]
sae_optimizer_name = ['adam']
sae_batch_size = [64]
sae_lambda_sparse = [0.1] #[0.1,0.5,2,5] # 0.1
sae_expansion_factor = [4] # 8
dataset_name = ['tiny_imagenet'] # cifar_10, mnist
'''

#'''
# local, ImageNet (among others, can be used for computing evaluation metrics)
sae_layer = ["mixed3a"]
model_name = ['inceptionv1']
directory_path = ['C:\\Users\\Jasper\\Downloads\\Master thesis\\Code']
wandb_status = ['0']
model_epochs = [1] #7
model_learning_rate = [0.001]
batch_size = [512]
model_optimizer_name = ['sgd']

sae_epochs = [61] 
sae_learning_rate = [0.001]
sae_optimizer_name = ['constrained_adam']
sae_batch_size = [256]
sae_lambda_sparse = [0.1] 
sae_expansion_factor = [2] #*7
dataset_name = ['imagenet']
sae_checkpoint_epoch = [35] #,37,40,42,45,47,50] #[3, 7, 15, 18, 20] # 0 means no checkpoint
#'''

#'''
# cluster, ImageNet
#sae_layer = ["mixed3b", "mixed4a", "mixed4b", "mixed4c", "mixed4d", "mixed4e", "mixed5a", "mixed5b"]
sae_layer = ["mixed3a"]
#sae_layer = ["mixed3b"]
#sae_layer = ["mixed4a"]
#sae_layer = ["mixed4b"]
#sae_layer = ["mixed4c"]
#sae_layer = ["mixed4d"]
#sae_layer = ["mixed4e"]
#sae_layer = ["mixed5a"]
#sae_layer = ["mixed5b"]

#sae_layer = ["mixed3b_3x3_pre_relu_conv"]
model_name = ['inceptionv1']
directory_path = ['/lustre/home/jtoussaint/master_thesis/']
wandb_status = ['0']
model_epochs = [1] #7
model_learning_rate = [0.001]
batch_size = [512]
model_optimizer_name = ['sgd']

sae_epochs = [13] #[40] #[10] #5
sae_learning_rate = [0.001] #[0.001, 0.0001, 0.00001]
sae_optimizer_name = ['constrained_adam']
sae_batch_size = [256] #[256,512,1024]
sae_lambda_sparse = [5.0] #[0.1,5.0,100.0] #[100] #[0.1, 1, 5, 10] #[0.1, 1, 5, 10] #[100] #[0.1, 0.5, 1,2,5,10]#[1e-1] 
sae_expansion_factor = [8] #[4,8,16] #[4,8,16]# [2,4,8] #[2,4,8,16] #[16] #2
dataset_name = ['imagenet']
sae_checkpoint_epoch = [0]*20 #,1,2,3,4,5,6,7,8,9,10]#,11,12] #[2,4,6,8] #,20,30,40,50]#*54#,30,40,50] #[1,3,5,7,10] #[20]*15#*20 #16 # 0 means no checkpoint
# we can also do [0,3,0,1] --> sae_2 uses checkpoint at epoch 3, sae_4 uses checkpoint at epoch 1
# order of models is according to itertools.product below: itertools.product(l1, l2,...)
# --> (l1[0], l2[0], ...), (l1[0], l2[1], ...), (l1[0], l2[2], ...), ..., (l1[1], l2[0], ...), ...
#'''

################################################################################################
# DON'T CHANGE ANYTHING BELOW THIS LINE

training = ['True'] if training == ['1'] else ['False']
original_model = ['True'] if original_model == ['1'] else ['False']

# Some checks whether parameters are set correctly
if mis != ["0"] or compute_ie != ['0']:
    if training == ["True"]:
        raise ValueError("If we do training, then nothing related to MIS or IE should be computed.")
    if dataset_name != ["imagenet"]:
        raise ValueError("MIS or IE can only be computed for the ImageNet dataset.")
if mis != ["0"] and compute_ie != ['0']:
    raise ValueError("Either compute MIS or IE, not both.")
if compute_ie != ['0'] and original_model == ['True']:
    raise ValueError("IE can only be computed for the SAE model, not the original model.")

# Generate all possible combinations of parameters. However, if we only want to train or evaluate the 
# original model, we only need to iterate over the parameters relevant for the original model.
# We need to modify this file since the jobs on the cluster are run based on the txt file generated
# by this script, i.e., if we generate 6 combinations, 6 jobs will be run on the cluster.
if original_model == ['True']:
    sae_model_name = ['None']
    sae_epochs = ['0']
    sae_learning_rate = ['0.0']
    sae_optimizer_name = ['None']
    sae_batch_size = ['0']
    sae_lambda_sparse = ['0']
    sae_expansion_factor = ['1'] # we computing the sparsity we divide by this number, so it should be 1 if we don't use SAE

if original_model == ['True'] and mis == ['0'] and compute_ie == ['0']:
    # as soon as mis != ['0'] (we compute MIS) or compute_ie != ['0'] (we compute IE), 
    # we need to specify the layer name, we misuse the sae_layer parameter for this purpose to refer to a layer of the original model
    sae_layer = ['None']

iters = [model_name,
        sae_model_name,
        sae_layer,
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
        dataset_name,
        training,
        original_model,
        model_criterion_name,
        sae_criterion_name,
        dead_neurons_steps,
        mis,
        compute_ie]

all_combinations = itertools.product(*iters)
number_combinations = functools.reduce(operator.mul, map(len, iters), 1)

if original_model == ['True']:
    sae_checkpoint_epoch = ['0'] * number_combinations
else:
    if len(sae_checkpoint_epoch) != number_combinations:
        raise ValueError('The number of elements in sae_checkpoint_epoch does not match the number of combinations.')

# Write the combinations to a text file which will be refered to
# by the .sh script file (if running the job on the cluster) and 
# the main.py file (if running the job locally)
number_combinations = 0
with open('parameters.txt', 'w') as file:
    for combination in all_combinations:
        line = ','.join(map(str, combination))+','+str(sae_checkpoint_epoch[number_combinations])
        file.write(line + '\n')
        number_combinations += 1

print(f'{number_combinations} parameter combinations written to parameters.txt')

# Now we create another such txt file for evaluating some results.
all_combinations_eval = itertools.product(model_name,
                                          sae_model_name,
                                            sae_layer, 
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
                                            dataset_name,
                                            original_model, 
                                            dead_neurons_steps,
                                            sae_checkpoint_epoch)
# We write parameters to a separate text file which will be used
# for computing some evaluation metrics.
number_combinations = 0
with open('parameters_eval.txt', 'w') as file_eval:
    for combination in all_combinations_eval:
        line = ','.join(map(str, combination))
        file_eval.write(line + '\n')
        number_combinations += 1

print(f'{number_combinations} parameter combinations written to parameters_eval.txt')