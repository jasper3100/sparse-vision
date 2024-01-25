import torch 
import os

from utils import store_feature_maps, get_module_names

class ActivationsHandler:
    '''
    Store the activations/feature maps of a selection of layers of the model.
    If desired, the output of a specific layer can be modified, in particular, by
    passing it through a sparse autoencoder (SAE) and then using the output of the
    SAE as the output of the layer.
    '''
    # constructor of the class (__init__ method)
    def __init__(self, 
                 model,
                 train_dataloader,
                 layer_name, 
                 dataset_name, 
                 original_activations_folder_path, 
                 adjusted_activations_folder_path=None,
                 sae_model=None):
        self.model = model
        self.train_dataloader = train_dataloader
        self.layer_name = layer_name
        self.dataset_name = dataset_name
        self.original_activations_folder_path = original_activations_folder_path
        self.adjusted_activations_folder_path = adjusted_activations_folder_path
        self.sae_model = sae_model
        self.activations = {} # placeholder to store activations
       
    def hook(self, module, input, output, name):
        # modify output of the specified layer if an sae_model is provided
        if self.sae_model is not None and name == self.layer_name:
            # input to the model = intermediate feature map outputted by a certain layer of the base model
            # shape of input to the model: [channels, height, width] --> no batch dimension
            _, out = self.sae_model(output[0]) # sae_model returns encoder_output, decoder_output
            if not torch.equal(out, output[0]):
                print(f"Successfully modified output of layer {name} for one batch of data")
            output[0] = out
        else: 
            pass
        # store the activations
        if name not in self.activations:
            self.activations[name] = []
        self.activations[name].append(output)

    def register_hooks(self):
        module_names = get_module_names(self.model)
        for name in module_names:
            m = getattr(self.model, name)
            m.register_forward_hook(lambda module, inp, out, name=name: self.hook(module, inp, out, name))
        # The below line works successfully for ResNet50:
        #self.model.layer1[0].conv3.register_forward_hook(lambda module, inp, out, name='model.layer1[0].conv3': self.hook(module, inp, out, name))
        # if I do this in ResNet50: m = getattr(module, 'layer1[0].conv3') --> not an attribute of the model

    def forward_pass(self):
        # Once we perform the forward pass, the hook will store the activations 
        # (and modify the output of the specified layer if desired)
        # As we iterate over batches, the activations will be appended to the dictionary
        batch_idx = 0
        self.register_hooks() # registering the hook within the for loop will lead to undesired behavior
        # as the hook will be registered multiple times --> activations will be captured multiple times!
        with torch.no_grad():
            if self.dataset_name == 'tiny_imagenet':
                for batch in self.train_dataloader:
                    inputs, _ = batch['image'], batch['label']  
                    self.model(inputs)
                    batch_idx += 1
                    if batch_idx == 2:
                        break # only do 2 batches for now
            elif self.dataset_name == 'cifar_10':
                for batch in self.train_dataloader:
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
        if self.sae_model is not None:
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

        if self.sae_model is not None:
            print(f"Successfully stored modified features with shape {combined_activations['fc1'].shape} for layer {self.layer_name}")
        else:
            print(f"Successfully stored original features with shape {combined_activations['fc1'].shape} for layer {self.layer_name}")

    def save_batch_num(self, num_batches):
        # if we consider the modified activations, we store them in a different location
        if self.sae_model is not None:
            folder_path = self.adjusted_activations_folder_path
        else:
            folder_path = self.original_activations_folder_path
        file_path = os.path.join(folder_path, 'num_batches.txt')
        with open(file_path, 'w') as f:
            f.write(str(num_batches))