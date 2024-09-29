from utils import *

sae_checkpoint_epoch = '6'
epoch_batch_idx = '5002' # just check the filename of the desired file
model_name = 'inceptionv1'
layer_name = "mixed3a"

# just for getting the right file name
model_epochs = '1'
model_learning_rate = '0.001'
batch_size = '512'
model_optimizer_name = 'sgd'

sae_model_name = 'sae_mlp'
sae_epochs = '10' # will be dropped anyways
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
sae_params_temp_copy = sae_params_temp.copy()
# for checkpointing, we consider all params apart from the number of epochs, because we will checkpoint at specific custom epochs
sae_params_temp.pop('sae_epochs', None)
params_string_sae_checkpoint = '_'.join(model_params_temp.values()) + "_" + "_".join(sae_params_temp.values())
params_string_sae_checkpoint_1 = '_'.join(model_params_temp.values()) + "_" + "_".join(sae_params_temp_copy.values())

folder_path = r"C:\Users\Jasper\Downloads\Master thesis\code\evaluation_results\inceptionv1\imagenet\indices_of_dead_neurons"
file_path = get_file_path(folder_path, layer_name, params_string_sae_checkpoint, f'epoch_{sae_checkpoint_epoch}_epoch_batch_idx_{epoch_batch_idx}.txt')
dead_units_indices_path = os.path.join(folder_path, file_path)
dead_units_indices = np.loadtxt(dead_units_indices_path, dtype=int)

mis_folder_path = r"C:\Users\Jasper\Downloads\Master thesis\code\evaluation_results\inceptionv1\imagenet\mis"
model_key = "sae"
file_path = get_file_path(folder_path=mis_folder_path, sae_layer=model_key + '_' + layer_name,params=params_string_sae_checkpoint_1,file_name=f'mis_epoch_{sae_checkpoint_epoch}.csv')
mis_path = os.path.join(mis_folder_path, file_path)
mis = pd.read_csv(mis_path)

# I want to select all rows where layer_name is in the column "layer_name" and "bottleneck" is not in the column "layer_name"
mis = mis[mis['layer_name'].str.contains(layer_name) & ~mis['layer_name'].str.contains("bottleneck")]

number_rows_before = len(mis)
average_mis_before = mis['MIS_confidence'].mean()
median_mis_before = mis['MIS_confidence'].median()

mis_dead_units = mis.iloc[dead_units_indices]
mis = mis.drop(dead_units_indices) # remove mis values of dead units
number_rows_after = len(mis)
diff_number_rows = number_rows_before - number_rows_after
if diff_number_rows != len(dead_units_indices):
    print(f"Warning: Number of removed rows ({diff_number_rows}) does not match the number of dead units ({len(dead_units_indices)}).")
average_mis_after = mis['MIS_confidence'].mean()
median_mis_after = mis['MIS_confidence'].median()

average_mis_dead_units = mis_dead_units['MIS_confidence'].mean()
median_mis_dead_units = mis_dead_units['MIS_confidence'].median()


print(dead_units_indices)

print("Number of dead units:", len(dead_units_indices))
print("-----------------")
print(f"Average MIS before: {average_mis_before}")
print(f"Median MIS before: {median_mis_before}")
print("-----------------")
print(f"Average MIS after: {average_mis_after}")
print(f"Median MIS after: {median_mis_after}")
print("-----------------")
print(f"Average MIS dead units: {average_mis_dead_units}")
print(f"Median MIS dead units: {median_mis_dead_units}")

# epoch 4, 138 dead units, no real difference
'''
Number of dead units: 138
-----------------
Average MIS before: 0.8599165960290436
Median MIS before: 0.865381655617748
-----------------
Average MIS after: 0.8593789835282453
Median MIS after: 0.865381655617748

'''
# epoch 6
'''
Number of dead units: 696
-----------------
Average MIS before: 0.86059835524283
Median MIS before: 0.867338594591743
-----------------
Average MIS after: 0.8570577305029466
Median MIS after: 0.8674655542630441
'''

# epoch 9
'''
Number of dead units: 604
-----------------
Average MIS before: 0.8462833886557861
Median MIS before: 0.8524580312588169
-----------------
Average MIS after: 0.8490719486967546
Median MIS after: 0.8585870771380294
'''