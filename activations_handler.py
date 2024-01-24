import torch 
import os

from utils import load_model_aux, load_data_aux
from utils_names import load_module_names
from utils_feature_map import store_feature_maps

class ActivationsHandler:
    '''
    Store the activations/feature maps of a selection of layers of the model.
    If desired, the output of a specific layer can be modified, in particular, by
    passing it through a sparse autoencoder (SAE) and then using the output of the
    SAE as the output of the layer.
    '''
    # constructor of the class (__init__ method)
    def __init__(self, 
                 model_name, 
                 layer_name, 
                 dataset_name, 
                 original_activations_folder_path, 
                 sae_weights_folder_path,
                 adjusted_activations_folder_path=None,
                 sae_model_name=None,
                 sae_dataset_name=None,
                 modify_output=False, 
                 expansion_factor=None,
                 batch_size=32):
        self.model_name = model_name
        self.sae_model_name = sae_model_name
        self.layer_name = layer_name
        self.dataset_name = dataset_name
        self.sae_dataset_name = sae_dataset_name
        self.original_activations_folder_path = original_activations_folder_path
        self.adjusted_activations_folder_path = adjusted_activations_folder_path
        self.sae_weights_folder_path = sae_weights_folder_path
        self.activations = {} # placeholder to store activations
        self.modify_output = modify_output
        self.expansion_factor = expansion_factor
        self.batch_size = batch_size
        self.training_data, _, self.img_size, _ = load_data_aux(self.dataset_name,
                                                                self.batch_size,
                                                                data_dir=None,
                                                                layer_name=self.layer_name)
        self.model, _ = load_model_aux(model_name, 
                                       self.img_size, 
                                       self.expansion_factor)
        self.model.eval()
        self.module_names = load_module_names(self.model_name, self.dataset_name, self.layer_name)

    def modify_layer_output(self, layer_output): 
        # layer_output = intermediate feature map outputted by the respective layer
        # shape of layer_output: [channels, height, width] --> no batch dimension
        if self.modify_output:
            # Instantiate the sparse autoencoder (SAE) model and load trained weights
            _, _, self.sae_img_size, _ = load_data_aux(self.sae_dataset_name,
                                                        self.batch_size,
                                                        data_dir=self.original_activations_folder_path,
                                                        layer_name=self.layer_name)
            sae, _ = load_model_aux(self.sae_model_name, 
                                    self.sae_img_size, 
                                    self.expansion_factor)
            sae_weights_file_path = os.path.join(self.sae_weights_folder_path, f'{self.layer_name}_model_weights.pth')
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
                print(f"Successfully modified output of layer {name} for one batch of data")
            output[0] = out

        # store the activations
        if name not in self.activations:
            self.activations[name] = []
        self.activations[name].append(output)

    def register_hooks(self):
        # print the layer names
        # print(f"Store activations step. Print layer names: {self.module_names}")

        for name in self.module_names:
            m = getattr(self.model, name)
            m.register_forward_hook(lambda module, inp, out, name=name: self.hook(module, inp, out, name))

        # The below line works successfully for ResNet50:
        #self.model.layer1[0].conv3.register_forward_hook(lambda module, inp, out, name='model.layer1[0].conv3': self.hook(module, inp, out, name))
        # if I do this in ResNet50: m = getattr(module, 'layer1[0].conv3') --> not an attribute of the model

        # This is what I did before, but it doesn't work anymore within classes etc + using 
        # exec is not recommended
        #for name in module_names:
        #    exec(f"{name}.register_forward_hook(lambda module, inp, out, name=name: self.hook(module, inp, out, name))")

    def forward_pass(self):
        # Once we perform the forward pass, the hook will store the activations 
        # (and modify the output of the specified layer if desired)
        # As we iterate over batches, the activations will be appended to the dictionary
        batch_idx = 0
        self.register_hooks() # registering the hook within the for loop will lead to undesired behavior
        # as the hook will be registered multiple times --> activations will be captured multiple times!
        with torch.no_grad():
            if self.dataset_name == 'sample_data_1':
                self.model(self.data)
            elif self.dataset_name == 'tiny_imagenet':
                for batch in self.training_data:
                    inputs, _ = batch['image'], batch['label']  
                    batch_idx += 1
                    self.model(inputs)
            elif self.dataset_name == 'cifar_10':
                for batch in self.training_data:
                    inputs, _ = batch
                    self.model(inputs)
                    batch_idx += 1
                    if batch_idx == 2:
                        break # only do 2 batches for now
            else:
                raise ValueError(f"Unsupported dataset: {self.dataset_name}")
        
        num_batches = batch_idx
        self.save_batch_num(num_batches)

    def save_activations(self):
        # if we consider the modified activations, we store them in a different location
        if self.modify_output:
            folder_path = self.adjusted_activations_folder_path
        else:
            folder_path = self.original_activations_folder_path
            
        # Combine activations from all batches
        combined_activations = {}
        for layer_name, activations_list in self.activations.items():
            combined_activations[layer_name] = torch.cat(activations_list, dim=0)

        store_feature_maps(combined_activations.keys(), # these are the module/layer names
                            combined_activations, 
                            folder_path)

        if self.modify_output:
            print(f"Successfully stored modified features with shape {combined_activations['fc1'].shape} for layer {self.layer_name}")
        else:
            print(f"Successfully stored original features with shape {combined_activations['fc1'].shape} for layer {self.layer_name}")

    def save_batch_num(self, num_batches):
        # if we consider the modified activations, we store them in a different location
        if self.modify_output:
            folder_path = self.adjusted_activations_folder_path
        else:
            folder_path = self.original_activations_folder_path
        file_path = os.path.join(folder_path, 'num_batches.txt')
        with open(file_path, 'w') as f:
            f.write(str(num_batches))


'''
if __name__ == "__main__":
    # Example usage
    model = # your model instantiation code
    layer_name = 'model.layer1[0].conv3'
    dataset_name = 'tiny_imagenet'
    activations_folder_path = # specify your path
    modify_output = True
    expansion_factor = 2

    activations_handler = ActivationsHandler(model, layer_name, dataset_name, activations_folder_path, modify_output, expansion_factor)
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