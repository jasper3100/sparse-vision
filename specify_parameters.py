import itertools

# Specify here which parameters to iterate over

steps_to_execute = [4,5]
'''
Step 1: train the model
Step 2: evalute model
Step 3: store activations
Step 4: train the SAE
Step 5: store modified activations
Step 6: evaluate modified model
'''
model_epochs = [30] #[1, 2]
model_learning_rate = [0.1] #[0.1, 0.2]
batch_size = [32]
model_optimizer = ['sgd']
sae_epochs = [5]#[15] #[1, 2]
sae_learning_rate = [0.001]
sae_batch_size = [64]
sae_optimizer = ['adam']
sae_lambda_sparse = [0.1, 0.5, 1.0, 2.0]
sae_expansion_factor = [2,4,6,8]
activation_threshold = [0.1]

# Concatenate the list of numbers as strings and convert it to an integer
# then we only have one argument specifying which steps to execute
steps_to_execute = [''.join(map(str, steps_to_execute))]

# Generate all possible combinations
all_combinations = itertools.product(
    steps_to_execute, model_epochs, model_learning_rate, batch_size, model_optimizer,
    sae_epochs, sae_learning_rate, sae_batch_size, sae_optimizer, sae_lambda_sparse, sae_expansion_factor, activation_threshold
)

# Write the combinations to a text file which will be refered to
# by the .sh script file
with open('parameters.txt', 'w') as file:
    for combination in all_combinations:
        line = ','.join(map(str, combination))
        file.write(line + '\n')