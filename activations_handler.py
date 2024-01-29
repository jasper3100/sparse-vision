import torch 

from utils import store_feature_maps, get_module_names, measure_sparsity, save_number

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
                 device,
                 train_dataloader,
                 layer_name, 
                 dataset_name, 
                 folder_path, 
                 eval_sparsity_threshold,
                 sae_model=None):
        self.model = model
        self.device = device
        self.train_dataloader = train_dataloader
        self.layer_name = layer_name
        self.dataset_name = dataset_name
        self.folder_path = folder_path
        self.eval_sparsity_threshold = eval_sparsity_threshold
        self.sae_model = sae_model

        self.activations = {} # placeholder to store activations
        self.activated_units = 0
        self.total_units = 0
       
    def hook(self, module, input, output, name):
        # modify output of the specified layer if an sae_model is provided
        if self.sae_model is not None and name == self.layer_name:
            # input to the model = intermediate feature map outputted by a certain layer of the base model
            # shape of input to the model: [channels, height, width] --> no batch dimension
            encoder_output, out = self.sae_model(output) # sae_model returns encoder_output, decoder_output

            # we measure the sparsity of the higher-dim. layer of the SAE, which shall be sparse
            activated_units, total_units = measure_sparsity(encoder_output, self.eval_sparsity_threshold)
            self.activated_units += activated_units
            self.total_units += total_units

            #if not torch.equal(out, output):
            #    print(f"Successfully modified output of layer {name} for one batch of data")
            output = out
        elif self.sae_model is None and name == self.layer_name: 
            # When passing the data through the original model we measure the sparsity of the output of the
            # specified layer so that we can compare it with the sparsity of the SAE higher dim. layer
            activated_units, total_units = measure_sparsity(output, self.eval_sparsity_threshold)
            self.activated_units += activated_units
            self.total_units += total_units
        else:
            pass

        # store the activations
        if name not in self.activations:
            self.activations[name] = output 
        else:
            self.activations[name] = torch.cat((self.activations[name], output), dim=0)
        
        return output

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
                    inputs = inputs.to(self.device)
                    self.model(inputs)
                    batch_idx += 1
                    #if batch_idx == 2:
                    #    break
            elif self.dataset_name == 'cifar_10':
                for batch in self.train_dataloader:
                    inputs, _ = batch
                    inputs = inputs.to(self.device)
                    self.model(inputs)
                    batch_idx += 1
                    #if batch_idx == 2:
                    #    break
            else:
                raise ValueError(f"Unsupported dataset: {self.dataset_name}")
        # Storing the number of batches is useful if we want to use less batches than the 
        # total number of batches for debugging purposes
        num_batches = batch_idx
        save_number(num_batches, self.folder_path, 'num_batches.txt')

        sparsity = 1 - self.activated_units / self.total_units
        save_number(sparsity, self.folder_path, f'{self.layer_name}_sparsity.txt')

    def save_activations(self):
        store_feature_maps(self.activations, self.folder_path)

        if self.sae_model is not None:
            print(f"Successfully stored modified features. In particular, the features of layer {self.layer_name} with shape {self.activations[self.layer_name].shape}")
        else:
            print(f"Successfully stored original features. In particular, the features of layer {self.layer_name} with shape {self.activations[self.layer_name].shape}")