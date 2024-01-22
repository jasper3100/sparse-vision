import torch 
import os

from get_module_names import ModuleNames
from sae import SparseAutoencoder
from utils import load_model, load_data, store_feature_maps
from functools import reduce

'''
Store the activations/feature maps of a selection of layers of the model.
If desired, the output of a specific layer can be modified, in particular, by
passing it through a sparse autoencoder (SAE) and then using the output of the
SAE as the output of the layer.
'''

#from main import parse_arguments
#args = parse_arguments()
#layer_name = args.layer_name
#dataset_name = args.dataset_name

class ActivationsHandler:
    # constructor of the class (__init__ method)
    def __init__(self, 
                 model_name, 
                 layer_name, 
                 dataset_name, 
                 original_activations_folder_path, 
                 sae_weights_folder_path,
                 modify_output=False, 
                 expansion_factor=None):
        self.model_name = model_name
        self.layer_name = layer_name
        self.dataset_name = dataset_name
        self.original_activations_folder_path = original_activations_folder_path
        self.sae_weights_folder_path = sae_weights_folder_path
        self.activations = {} # placeholder to store activations
        self.modify_output = modify_output
        self.expansion_factor = expansion_factor
        self.model, _ = load_model(self.model_name)
        self.model.eval()
        self.data = load_data(self.dataset_name)

    def get_module_names(self):
        # get the names of the main modules of the model and include layer_name
        # This list of module names can be adjusted as desired, i.e., removing/adding layers
        '''
        If we want to use all layers after the specified layer, we can do 
        layer_names = module_names.get_names_of_all_layers()
        trunc_layer_names = layer_names[layer_names.index(self.layer_name):]    
        '''
        module_names = ModuleNames(self.model_name)
        return module_names.names_of_main_modules_and_specified_layer(self.layer_name)

    def modify_layer_output(self, layer_output): 
        # layer_output = intermediate feature map outputted by the respective layer
        # shape of layer_output: [channels, height, width] --> no batch dimension
        if self.modify_output:
            # Instantiate the sparse autoencoder (SAE) model and load trained weights
            sae = SparseAutoencoder(input_tensor=layer_output, expansion_factor=self.expansion_factor)
            sae_weights_file_path = os.path.join(self.sae_weights_folder_path, f'{self.layer_name}_trained_sae_weights.pth')
            sae.load_state_dict(torch.load(sae_weights_file_path))
            sae.eval()
            _, modified_output = sae(layer_output)
            return modified_output
        else:
            return layer_output


    def hook(self, module, input, output, name):
        # modify output of the specified layer if requested
        if name == self.layer_name:
            out = self.modify_layer_output(output[0])
            # if out and output[0] are different then print success message
            if not torch.equal(out, output[0]):
                print(f"Successfully modified output of layer {name}")
            output[0] = out

        # store the activations
        if name not in self.activations:
            self.activations[name] = []
        self.activations[name].append(output)

    def register_hooks(self):
        module_names = self.get_module_names()
        
        # The below line works successfully:
        #model.layer1[0].conv3'
        self.model.layer1[0].conv3.register_forward_hook(lambda module, inp, out, name='model.layer1[0].conv3': self.hook(module, inp, out, name))
        
        #module = self.model

        #attributes = [attr for attr in dir(self.model) if not callable(getattr(self.model, attr))]
        #print(attributes)

        #print(dir(self.model))

        # m = getattr(module, 'layer1[0].conv3') --> not an attribute of the model
        #m = getattr(module, 'layer1')
        
        #print(m)
        #m.register_forward_hook(lambda module, inp, out, name='model.layer1': self.hook(module, inp, out, name))

        #'''
        module_names = ModuleNames(self.model_name)
        layers = module_names.get_lowest_level_modules()
        #print(layers.keys())
        #print(layers.values())
        module = list(layers.values())[-5]
        name = list(layers.keys())[-5]
        print(name)
        print(module)
        module.register_forward_hook(lambda module, inp, out, name=name: self.hook(module, inp, out, name))
        #'''

        '''
        for name in module_names:
            for attr in name.split('.'):
                # Use getattr to dynamically access the attribute
                print(attr)
                #module = getattr(module, attr)
                #print(module)
        '''

        #module.register_forward_hook(lambda module, inp, out, name=name: self.hook(module, inp, out, name))


        #for name in module_names:
        #    exec(f"{name}.register_forward_hook(lambda module, inp, out, name=name: self.hook(module, inp, out, name))")
   
    '''
    def register_hooks(self):
        self.module_names = self.get_module_names()
        for name in self.module_names:
            if "[" in name and "]" in name:
                # This is a nested module
                module = self.model
                for attr in name.split('.'):
                    if "[" in attr and "]" in attr:
                        idx = int(attr.split('[')[1].split(']')[0])
                        module = module[idx]
                    else:
                        module = getattr(module, attr)
            else:
                # This is not a nested module
                module = reduce(getattr, name.split('.'), self.model)

            module.register_forward_hook(lambda module, inp, out, name=name: self.hook(module, inp, out, name))
    '''

    def forward_pass(self):
        # Once we perform the forward pass, the hook will store the activations 
        # (and possibly modify the output of the specified layer)
        with torch.no_grad():
            if self.dataset_name == 'sample_data_1':
                output = self.model(self.data)
            elif self.dataset_name == 'tiny_imagenet':
                for batch in self.data:
                    inputs, labels = batch['image'], batch['label']            
                    output = self.model(inputs)
                    print(output.shape)
                    print(labels.shape)
                    break # do only one batch for now

    def save_activations(self):
        store_feature_maps(self.activations.keys(), self.activations, self.original_activations_folder_path)
        # instead of self.activations.keys(), we can also use module_names (which would have to be extracted from above)



'''
if __name__ == "__main__":
    # Example usage
    model = # your model instantiation code
    layer_name = 'model.layer1[0].conv3'
    dataset_name = 'tiny_imagenet'
    original_activations_folder_path = # specify your path
    modify_output = True
    expansion_factor = 2

    activations_handler = ActivationsHandler(model, layer_name, dataset_name, original_activations_folder_path, modify_output, expansion_factor)
    activations_handler.register_hooks()

    # Replace 'data' with your actual input data
    data = # your input data
    activations_handler.forward_pass(data)

    activations_handler.save_activations()
'''


'''
for epoch in range(num_epochs):
    for batch in dataloader:
        inputs, labels = batch['image'], batch['label']

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
'''