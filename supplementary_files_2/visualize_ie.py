# visualize the IE of the SAE features
from utils import *

save_figures = True #True # set to True if you want to save the figures
show_figures = False # set to True if you want to display them

layers = ['mixed3a', 'mixed3b', 'mixed4b', 'mixed4c', 'mixed4d', 'mixed4e', 'mixed5a', 'mixed5b'] # skip mixed4a for now

evaluation_results_folder_path = r"C:\Users\Jasper\Downloads\Master thesis\code\evaluation_results\inceptionv1\imagenet"
ie_related_quantities = os.path.join(evaluation_results_folder_path, "ie_related_quantities")
IE_SAE_features_folder_path = os.path.join(ie_related_quantities, "IE_SAE_features")
IE_SAE_errors_folder_path = os.path.join(ie_related_quantities, "IE_SAE_errors")
IE_Model_neurons_folder_path = os.path.join(ie_related_quantities, "IE_Model_neurons")
IE_SAE_edges_folder_path = os.path.join(ie_related_quantities, "IE_SAE_edges")
circuit_dead_units_folder_path = os.path.join(ie_related_quantities, "dead_units") # specific to the circuit samples that we use
MIS_folder_path = os.path.join(evaluation_results_folder_path, "MIS")
#dead_units_folder_path = os.path.join(evaluation_results_folder_path, "indices_of_dead_neurons")
store_new_figures = r"C:\Users\Jasper\Downloads\Master thesis\code\evaluation_results\inceptionv1\imagenet\ie_plots_cpu"
# create the folder if it doesn't exist
if not os.path.exists(store_new_figures):
    os.makedirs(store_new_figures)

model_params = {'model_name': "inceptionv1", 'epochs': 1, 'learning_rate': 0.001, 'batch_size': 512, 'optimizer': 'sgd'}
sae_model_name = "sae_mlp"
sae_optimizer_name = "constrained_adam"
model_params_temp = {k: str(v) for k, v in model_params.items()}

params_string_ie = {}
sae_checkpoint_epoch = {}
params_string_mis = {}
params_string_1 = {}
for name in layers:
    params_string_ie[name], sae_checkpoint_epoch[name], _, params_string_mis[name], dead_neurons_steps = get_specific_sae_params(name, sae_model_name, model_params_temp, sae_optimizer_name)
    params_string_1[name] = '_'.join(model_params_temp.values()) + "_" + sae_model_name + "_" + str(dead_neurons_steps)

ie_sae_features = {}
ie_sae_error = {}
ie_model_neurons = {}
ie_edges = {}
mis_vals = {}
circuit_dead_units = {}
sparsity = {}
perc_dead_units = {}
#dead_units = {} # DO WE NEED THIS???
epoch_batch_idx = 5002

model_version = "torchvision"
#model_version = "lucent"

if model_version == "torchvision":
    file_ending = "torchvision.pt"
elif model_version == "lucent":
    file_ending = ".pt"

for name in layers:
    # file paths
    ie_sae_features_file_path = get_file_path(IE_SAE_features_folder_path, name, params_string_ie[name], file_ending)
    ie_sae_error_file_path = get_file_path(IE_SAE_errors_folder_path, name, params_string_ie[name], file_ending)
    ie_model_neurons_file_path = get_file_path(IE_Model_neurons_folder_path, name, params_string_ie[name], file_ending)
    ie_edges_file_path = get_file_path(IE_SAE_edges_folder_path, name, params_string_ie[name], '.pt')
    model_key = "sae"
    mis_file_path = get_file_path(folder_path=MIS_folder_path, sae_layer=model_key + '_' + name, params=params_string_mis[name], file_name=f'.csv')
    circuit_dead_units_file_path = get_file_path(circuit_dead_units_folder_path, name, params_string_ie[name], file_ending)
    # get the sparsity and perc of dead neurons of the SAEs on the full eval dataset
    sae_eval_results_path = get_file_path(folder_path=evaluation_results_folder_path, sae_layer=name, params=params_string_1[name], file_name='sae_eval_results.csv')
    #dead_units_path = get_file_path(dead_units_folder_path, name, params_string_ie[name], f'epoch_{sae_checkpoint_epoch[name]}_epoch_batch_idx_{epoch_batch_idx}.txt')
    
    # load the values
    ie_sae_features[name] = torch.load(ie_sae_features_file_path, map_location=torch.device('cpu')) # shape [C*K]
    ie_sae_error[name] = torch.load(ie_sae_error_file_path, map_location=torch.device("cpu")) # shape: scalar
    ie_model_neurons[name] = torch.load(ie_model_neurons_file_path, map_location=torch.device("cpu")) # shape: [C]
    circuit_dead_units[name] = torch.load(circuit_dead_units_file_path, map_location=torch.device("cpu")) # shape: [C*K], 
    #ie_edges[name] = torch.load(ie_edges_file_path) # shape: [num_sae_features_current_layer + 1, num_sae_features_next_layer + 1]
    mis = pd.read_csv(mis_file_path)
    mis_vals[name] = mis['MIS_confidence'].values # shape: [C*K]
    sae_eval_results_csv = pd.read_csv(sae_eval_results_path)
    sparsity[name] = sae_eval_results_csv['rel_sparsity'].values[sae_checkpoint_epoch[name]]
    perc_dead_units[name] = sae_eval_results_csv['perc_dead_units'].values[sae_checkpoint_epoch[name]]
    #with open(dead_units_path, 'r') as f:
    #    dead_units[name] = [int(line.strip()) for line in f]

#get the indices of ie_sae_features["mixed3a"] corresponding to values sorted from high to low
#sorted_indices = torch.argsort(ie_sae_features["mixed4c"], descending=True)
#torch.set_printoptions(profile="full") #threshold=100)
#print(sorted_indices)

# stop code here
#raise SystemExit

if model_version == "torchvision":
    circuit_sparsity_path = os.path.join(ie_related_quantities, "sparsity_torchvision.csv")         
    circuit_perc_dead_units_path = os.path.join(ie_related_quantities, "perc_dead_units_torchvision.csv")
elif model_version == "lucent":
    circuit_sparsity_path = os.path.join(ie_related_quantities, "sparsity.csv")
    circuit_perc_dead_units_path = os.path.join(ie_related_quantities, "perc_dead_units.csv")
circuit_sparsity_csv = pd.read_csv(circuit_sparsity_path)
circuit_perc_dead_units_csv = pd.read_csv(circuit_perc_dead_units_path)

###################################################
# PLOT SPARSITY OF UNITS IN CIRCUIT AND EVAL DATASET
plt.figure(figsize=(10, 5))
plt.plot(circuit_sparsity_csv['Layer'], circuit_sparsity_csv['Sparsity'], label='Sparsity circuit')
plt.plot(list(sparsity.keys()), list(sparsity.values()), label='Sparsity eval dataset')
plt.xlabel('Layer')
plt.ylabel('Sparsity')
plt.title('Sparsity of units in circuit vs eval dataset')
plt.legend(loc='upper right')
if save_figures:
    if model_version == "torchvision":
        # store the file in the folder "store_new_figures" and call it "sparsity_circuit_vs_eval_dataset_torchvision.png"
        plt.savefig(os.path.join(store_new_figures, 'sparsity_circuit_vs_eval_dataset_torchvision.png'))
    else:
        plt.savefig(os.path.join(store_new_figures, 'sparsity_circuit_vs_eval_dataset.png'))
if show_figures:
    plt.show()

###################################################
# PLOT PERC OF DEAD UNITS IN CIRCUIT AND EVAL DATASET
plt.figure(figsize=(10, 5))
plt.plot(circuit_perc_dead_units_csv['Layer'], circuit_perc_dead_units_csv['Percentage dead units'], label='Perc dead units circuit')
plt.plot(list(perc_dead_units.keys()), perc_dead_units.values(), label='Perc dead units eval dataset')
plt.xlabel('Layer')
plt.ylabel('Perc dead units')
plt.title('Perc of dead units in circuit vs eval dataset')
plt.legend(loc='upper right')
if save_figures:
    if model_version == "torchvision":
        plt.savefig(os.path.join(store_new_figures, 'perc_dead_units_circuit_vs_eval_dataset_torchvision.png'))
    else:
        plt.savefig(os.path.join(store_new_figures, 'perc_dead_units_circuit_vs_eval_dataset.png'))
if show_figures:
    plt.show()

###################################################
# REMOVE IE OF DEAD UNITS
# ciruit_dead_units: True if a unit is dead
#for name in layers:
#    print(f"Number of dead units in layer {name}: {torch.sum(circuit_dead_units[name])}")
#    print(len(ie_sae_features[name]))
#ie_sae_features = {name: ie_sae_features[name][~circuit_dead_units[name]] for name in layers}

#print(torch.tensor(list(ie_sae_error.values())))

###################################################
# HISTOGRAM OF IE VALUES OF SAE FEATURES/ERRORS BY LAYER
colors = plt.cm.tab10(np.linspace(0, 1, len(layers)))
plt.figure(figsize=(10, 6))
data = list(ie_sae_features.values())
#min_data = min([min(d) for d in data]).item()
#max_data = max([max(d) for d in data]).item()
# Define the bin edges
#bins_below = np.linspace(min_data, -1e-16, 10)
#bins_above = np.linspace(1e-16, max_data, 10)
#bin_zero = np.array([-1e-16, 1e-16])
#bins = np.concatenate((bins_below[:-1], bin_zero, bins_above[1:]))
plt.hist(data, bins=50, color=colors, edgecolor='black', label=layers, stacked=True)
plt.xlabel('IE value')
plt.ylabel('Frequency')
plt.title('Stacked Histogram of IE values of SAE features')
plt.legend(loc='upper right')
if save_figures:
    if model_version == "torchvision":
        plt.savefig(os.path.join(store_new_figures, 'histogram_ie_values_sae_features_torchvision.png'))
    else:
        plt.savefig(os.path.join(store_new_figures, 'histogram_ie_values_sae_features.png'))
if show_figures:
    plt.show()

'''
colors = plt.cm.tab10(np.linspace(0, 1, len(layers)))
plt.figure(figsize=(10, 6))
plt.hist(list(ie_edges.values()), bins=10, color=colors, edgecolor='black', label=layers, stacked=True)
plt.xlabel('IE value')
plt.ylabel('Frequency')
plt.title('Stacked Histogram of IE values of edges')
plt.legend(loc='upper right')
plt.show()
'''

###################################################
# PLOT MEDIAN IE VALUES OF SAE FEATURES/ERRORS BY LAYER
# for each layer compute the median IE value of SAE features and plot them

ie_median_sae = {}
ie_median_model = {}
for name in layers:
    ie_values_aux = torch.cat([ie_sae_features[name], torch.tensor([ie_sae_error[name]])])
    ie_values_aux_non_zero = ie_values_aux[ie_values_aux != 0]
    # if we don't remove zero values the median of layer 5b is zero, which looks weird in the plot
    ie_median_sae[name] = torch.median(ie_values_aux_non_zero).item()
    ie_median_model[name] = torch.median(ie_model_neurons[name][ie_model_neurons != 0]).item()
plt.figure(figsize=(10, 5))
# replace "mixed" by "" in the layer names
print(ie_median_sae["mixed5b"])
print(ie_sae_features["mixed5b"])
layers_aux = [name.replace("mixed", "") for name in layers]
plt.plot(layers_aux, list(ie_median_sae.values()), marker='o', label='SAE')
plt.plot(layers_aux, list(ie_median_model.values()), marker='o', label='Model')
fontsize=22
plt.xlabel('Layer', fontsize=fontsize, labelpad=5)
plt.yscale('log')
plt.ylabel('Median IE', fontsize=fontsize, labelpad=5)
plt.tick_params(axis='both', which='major', labelsize=fontsize - 2)
plt.legend(fontsize=fontsize - 2)
#plt.title('Median IE of SAE features per layer')
plt.tight_layout()
if save_figures:
    if model_version == "torchvision":
        plt.savefig(os.path.join(store_new_figures, 'median_ie_values_sae_features_torchvision.png'))
    else:
        plt.savefig(os.path.join(store_new_figures, 'median_ie_values_sae_features.png'))
if show_figures:
    plt.show()

# compute median IE value of edges from one layer to the next layer
'''
ie_edges_median = {}
for name in layers:
    ie_edges_median[name] = torch.median(ie_edges[name]).item()
plt.figure(figsize=(10, 5))
plt.plot(layers, list(ie_edges_median.values()), marker='o')
plt.xlabel('Upstream layer')
plt.ylabel('Median IE')
plt.title('Median IE of edges per layer')
plt.show()
'''

###################################################
# PLOT NUMBER OF NODES/EDGES IN CIRCUITS VS IE THRESHOLD
# Collect the IE values of all SAE features and all SAE error nodes
model_or_sae = "model"

if model_or_sae == "sae":
    ie_values = torch.cat([torch.cat(list(ie_sae_features.values())), torch.tensor(list(ie_sae_error.values()))])
elif model_or_sae == "model":
    ie_values = torch.cat(list(ie_model_neurons.values()))

# instead of the min take the min that is larger than 0, otherwise the thresholds are not computed correctly, because
# 0 is taken as the min and then we don't have thresholds in very small areas such as 10^-10
# hence, we simply remove the 0 values, if we remove dead neurons, we do not have 0 values (!)
ie_values = ie_values[ie_values != 0]
# Moreover, we get logarithmic thresholds so that we have more values in the lower range, otherwise we 
# skip a lot of values in the lower range
log_start = np.log10(ie_values.min())
log_stop = np.log10(ie_values.max())
log_values = np.linspace(log_start, log_stop, num=100)
#print(np.power(10, np.linspace(log_start, log_stop, num=20)))
# Convert back to the original scale
ie_thresholds = np.power(10, log_values)
# alternatively, linearly spaced thresholds:
#ie_thresholds = torch.linspace(ie_values.min(), ie_values.max(), 100)

# The number of nodes is the number of ie_values that are above the threshold for any of the below thresholds
num_nodes = [torch.sum(ie_values > threshold).item() for threshold in ie_thresholds]
# plot num_nodes vs ie_thresholds
plt.figure(figsize=(10, 5))
plt.plot(ie_thresholds, num_nodes)
fontsize=22
plt.xlabel('IE threshold', fontsize=fontsize, labelpad=5)
plt.xscale('log')
#plt.yscale('log')
#plt.ylim(0, 200)
#plt.xlim(1e-3, 1)
plt.ylabel('Number of nodes', fontsize=fontsize, labelpad=5)
plt.tick_params(axis='both', which='major', labelsize=fontsize - 2)
# clip 0 to the left bottom corner 
plt.xlim(left=0)
plt.ylim(bottom=0)
#plt.title('Number of nodes in circuits vs IE threshold')
if model_or_sae == "sae":
    name = "sae"
else:
    name = "model"
plt.tight_layout()

if save_figures:
    if model_version == "torchvision":
        plt.savefig(os.path.join(store_new_figures, name + '_' + 'num_nodes_vs_ie_threshold_torchvision.png'))
    else:
        plt.savefig(os.path.join(store_new_figures, name + '_' + 'num_nodes_vs_ie_threshold.png'))
if show_figures:
    plt.show()

# Collect the IE values of all edges
'''
ie_values = torch.cat(list(ie_edges.values()))
ie_thresholds = torch.linspace(ie_values.min(), ie_values.max(), 100)
num_edges = [torch.sum(ie_values > threshold).item() for threshold in ie_thresholds]
plt.figure(figsize=(10, 5))
plt.plot(ie_thresholds, num_edges)
plt.xlabel('IE threshold')
plt.ylabel('Number of edges')
plt.title('Number of edges in circuits vs IE threshold')
plt.show()
'''

# Find linearly separated threshold values 
#print(torch.linspace(ie_values.min(), ie_values.max(), 25))

###################################################
# GET THRESHOLD FOR IE VALUES
# we have 8 layers, for each one we probably need the SAE errors as nodes
# if we want 3 nodes per layer we would need 24 nodes + 8 SAE errors -> 32 nodes
# select threshold such that 32 values are above it -> the 33rd highest value is the threshold
sorted_ie_values = torch.sort(ie_values, descending=True).values
threshold = sorted_ie_values[32] # 33rd highest value (recall that we have 0-indexing)

#print(f"Threshold for IE values: {threshold}")
#print("Indices of neurons with IE > threshold:")
#for name in layers:
#    print(f"Layer {name}: {torch.nonzero(ie_sae_features[name] > threshold).flatten()}, SAE error: {ie_sae_error[name] > threshold}")

###################################################
# PLOT IE VS MIS (for each SAE features)
# we only consider SAE features as we don't compute MIS for SAE errors

'''
ie_mis = {}
colors = plt.cm.tab10(np.linspace(0, 1, len(layers)))
plt.figure(figsize=(10, 6))
plt.xlabel('IE')
plt.ylabel('MIS')
plt.title('IE vs MIS for SAE features')

# load the MIS values
for name in layers:
    # store the IE and MIS values of each node as a tuple
    ie_mis[name] = list(zip(ie_sae_features[name], mis_vals[name]))
    # add to the scatter plot in a specific color
    plt.scatter(*zip(*ie_mis[name]), color=colors[layers.index(name)], label=name)

plt.legend(loc='upper right')
#plt.show()
'''

###################################################
# PLOT IE THRESHOLD VS MEDIAN MIS (for SAE features)

'''
ie_values = torch.cat(list(ie_sae_features.values()))
ie_thresholds = torch.linspace(ie_values.min(), ie_values.max(), 100)
# for each threshold, compute the median MIS of SAE features
median_mis = [torch.median(torch.tensor([mis for ie, mis in ie_mis.values() if ie > threshold])).item() for threshold in ie_thresholds]
# for each threshold, compute number of nodes
plt.figure(figsize=(10, 5))
plt.plot(ie_thresholds, median_mis)
plt.xlabel('IE threshold')
plt.ylabel('Median MIS')
plt.title('Median MIS of SAE features vs IE threshold')
#plt.show()
'''

###################################################
# PLOT MEDIAN MIS VS NUMBER OF NODES FOR EACH THRESHOLD (for SAE features)

# We want as few nodes as possible and as high MIS as possible --> choose IE threshold accordingly

'''
num_nodes = [torch.sum(ie_values > threshold).item() for threshold in ie_thresholds]
plt.figure(figsize=(10, 5))
plt.plot(num_nodes, median_mis)
plt.xlabel('Number of nodes')
plt.ylabel('Median MIS')
plt.title('Median MIS of SAE features vs number of nodes')
#plt.show()
'''