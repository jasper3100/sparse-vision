import re
import os
import h5py

from model import model 
from data import input_data

'''
Auxiliary functions:
- print classification results of model for:
    - a batch of samples
    - a single sample
- print all layer names
- get main modules and names of model
- store activations/feature maps
'''



# Print result of a batch of samples
def print_result(predictions, weights):
    # Iterate over each sample in the batch
    for i in range(predictions.size(0)):
        prediction = predictions[i].softmax(0)
        class_id = prediction.argmax().item()
        score = prediction[class_id].item()
        category_name = weights.meta["categories"][class_id]
        print(f"Sample {i + 1}: {category_name}: {100 * score:.1f}%")



# Print result of a single sample
def print_result_sample(output, weights):
    prediction = output.squeeze(0).softmax(0)
    class_id = prediction.argmax().item()
    score = prediction[class_id].item()
    category_name = weights.meta["categories"][class_id]
    print(f"{category_name}: {100 * score:.1f}%")



# Print names of all layers of the model
'''
First, get names of all model layers (layers at the lowest level in case of nested submodules,
i.e. modules that do not contain any submodules. Because otherwise, f.e. using 
"module_names = [n for n, _ in model.named_modules()]" we will also retrieve, 
f.e., "layer1", which contains many modules/layers and is thus ambiguous. 
'''
def get_name_of_lowest_level_modules(model):
    lowest_level_modules = []
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:
            lowest_level_modules.append(name)
    return lowest_level_modules

def convert_name(name):
    # convert .1.2.3. to [1][2][3]
    name_brackets = re.sub(r'\.(\d+)', r'[\1]', name)
    # add model. to the beginning
    return f"model.{name_brackets}"

def get_names_of_all_layers(model):
    module_names = get_name_of_lowest_level_modules(model)
    # remove empty strings if any
    module_names = [name for name in module_names if name]
    return [convert_name(name) for name in module_names]

#layer_names = get_names_of_all_layers(model)
#print(layer_names)



def get_main_modules_and_names(model):
    names = []
    modules = []
    for name, module in model.named_children():
        names.append(name)
        modules.append(module)
    return names, modules

def names_of_main_modules_and_specified_layer(model, layer_name):
    # get the names of the main modules of the model
    module_names, _ = get_main_modules_and_names(model)
    # Add "model." to the beginning of each name
    module_names = [f"model.{name}" for name in module_names]
    # append layer_name to module_names
    module_names.append(layer_name)
    return module_names
    



def store_feature_maps(layer_names, activations, folder_path):
    # store the intermediate feature maps
    for name in layer_names:
        activation = activations[name][0] # we do [0], because the tensor that we want is inside of a list
        
        # Ensure the folder exists; create it if it doesn't
        os.makedirs(folder_path, exist_ok=True)

        activations_file_path = os.path.join(folder_path, f'{name}_activations.h5')
        
        # Store activations to an HDF5 file
        with h5py.File(activations_file_path, 'w') as h5_file:
            h5_file.create_dataset('data', data=activation.numpy())

    print("Successfully stored features in", folder_path)