import itertools

from old_main_with_individual_steps import execute_project, get_vars, parse_arguments

# Example command line:
# python gridsearch_main.py --model_name custom_mlp_1 --dataset_name cifar_10 --layer_name fc1 --directory_path '/lustre/home/jtoussaint/master_thesis/'

# SPECIFY HERE WHICH PARAMETERS TO ITERATE OVER

steps_to_execute = [1,2]
'''
Step 1: train the model
Step 2: evalute model
Step 3: store activations
Step 4: train the SAE
Step 5: store modified activations
Step 6: evaluate modified model
'''
model_epochs = [5,10] #[1, 2]
model_learning_rate = [0.1, 0.001] #[0.1, 0.2]
batch_size = [32, 64]
model_optimizer = ['sgd', 'adam']
sae_epochs = [1] #[1, 2]
sae_learning_rate = [0.1, 0.2]
sae_batch_size = [32]
sae_optimizer = ['adam']
    
if __name__ == '__main__':
    args = parse_arguments(name='gridsearch')
    variables = get_vars(args,name='gridsearch')
    
    # Concatenate the list of numbers as strings and convert it to an integer
    # then we only have one argument specifying which steps to execute
    steps_to_execute = [''.join(map(str, steps_to_execute))]

    # Generate all possible combinations
    all_combinations = itertools.product(
        steps_to_execute, model_epochs, model_learning_rate, model_optimizer,
        sae_epochs, sae_learning_rate, sae_optimizer, batch_size, sae_batch_size
    )

    for iter_variables in all_combinations:
        execute_project(*variables, *iter_variables)
    # THE ORDER OF INPUT ARGUMENTS IS IMPORTANT! Hence, when changing something, pay attention that the order of input arguments is still the same