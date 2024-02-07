import torch 

from utils import *

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
                 layer_names, 
                 dataset_name, 
                 folder_path, 
                 activation_threshold,
                 params,
                 sae_model=None,
                 encoder_output_folder_path=None,
                 dataset_length=None,
                 prof=None):
        self.model = model
        self.device = device
        self.train_dataloader = train_dataloader
        self.layer_names = layer_names
        self.dataset_name = dataset_name
        self.folder_path = folder_path
        self.activation_threshold = activation_threshold
        self.params = params
        self.sae_model = sae_model
        self.encoder_output_folder_path = encoder_output_folder_path
        self.dataset_length = dataset_length
        self.prof = prof

        self.activations = {} # placeholder to store activations
        self.sparsity = {} # placeholder to store sparsity values
        self.encoder_output = {} # placeholder to store encoder output of SAE
       
    def hook(self, module, input, output, name):
        # modify output of a layer if an sae_model is provided
        if self.sae_model is not None and name in self.layer_names:
            # input to the model = intermediate feature map outputted by a certain layer of the base model
            # shape of input to the model: [channels, height, width] --> no batch dimension
            encoder_output, decoder_output = self.sae_model(output) # sae_model returns encoder_output, decoder_output

            # we measure the sparsity of the higher-dim. layer of the SAE, which shall be sparse
            activated_units, total_units = measure_activating_units(encoder_output, self.activation_threshold) 

            # we store the encoder output for later analysis
            if name not in self.encoder_output:
                self.encoder_output[name] = encoder_output
            else:
                self.encoder_output[name] = torch.cat((self.encoder_output[name], encoder_output), dim=0)

            #if not torch.equal(out, output):
            #    print(f"Successfully modified output of layer {name} for one batch of data")
            output = decoder_output
        elif self.sae_model is None and name in self.layer_names: 
            # When passing the data through the original model we measure the sparsity of the output of the
            # specified layer so that we can compare it with the sparsity of the SAE higher dim. layer
            activated_units, total_units = measure_activating_units(output, self.activation_threshold)
        else:
            pass

        # store the activations of layer 'name' for the current batch, this takes 39 seconds for MNIST
        #store_batch_feature_maps(output, self.dataset_length, name, self.folder_path, self.params)
        ''' # the cat operation takes a long time, overall 93 seconds for MNIST
        if name not in self.activations:
            self.activations[name] = output 
        else:
            self.activations[name] = torch.cat((self.activations[name], output), dim=0)
        '''

        # store the activations, this takes 28 seconds for MNIST
        if name not in self.activations:
            self.activations[name] = []
        self.activations[name].append(output)

        # store the sparsity info of all layers in the provided list of layer names
        if name in self.layer_names:
            if name not in self.sparsity:
                self.sparsity[name] = (activated_units, total_units)
            else:
                self.sparsity[name] = (self.sparsity[name][0] + activated_units, self.sparsity[name][1] + total_units)
            
        return output

    def register_hooks(self):
        module_names = get_module_names(self.model)
        for name in module_names:
            m = getattr(self.model, name)
            m.register_forward_hook(lambda module, inp, out, name=name: self.hook(module, inp, out, name))
        # The below line works successfully for ResNet50:
        #self.model.layer1[0].conv3.register_forward_hook(lambda module, inp, out, name='model.layer1[0].conv3': self.hook(module, inp, out, name))
        # if I do this in ResNet50: m = getattr(module, 'layer1[0].conv3') --> not an attribute of the model

    def forward_pass(self, num_batches=None):
        # Once we perform the forward pass, the hook will store the activations 
        # (and modify the output of the specified layer if desired)
        # As we iterate over batches, the activations will be appended to the dictionary
        batch_idx = 0
        self.register_hooks() # registering the hook within the for loop will lead to undesired behavior
        # as the hook will be registered multiple times --> activations will be captured multiple times!
        with torch.no_grad():
            for batch in self.train_dataloader:
                if isinstance(batch, (list, tuple)) and len(batch) == 2:
                    inputs, _ = batch
                    #print(inputs.shape, targets.shape)
                elif isinstance(inputs, torch.Tensor):
                    # if the dataloader doesn't contain targets, then we use
                    # the inputs as targets (f.e. autoencoder reconstruction loss)
                    pass
                else:
                    raise ValueError("Unexpected data format from dataloader")
                
                inputs = inputs.to(self.device)
                self.model(inputs)
                batch_idx += 1
                # do a profiler step if a profiler is provided
                if self.prof is not None:
                    self.prof.step()
                if batch_idx == num_batches:
                    break
            ''' # Alternatively
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
            '''
        # store the sparsity information
        for name in self.sparsity.keys():
            activated_units, total_units = self.sparsity[name]
            file_path = get_file_path(folder_path=self.folder_path, 
                                    layer_names=[name], 
                                    params=self.params, 
                                    file_name='sparsity.txt')
            save_numbers((activated_units, total_units), file_path)

    def save_activations(self):

        for name in self.activations.keys():
            self.activations[name] = torch.cat(self.activations[name], dim=0)

        store_feature_maps(self.activations, self.folder_path, params=self.params)
        # also save the encoder output of the SAE here
        store_feature_maps(self.encoder_output, self.encoder_output_folder_path, params=self.params)

        if self.sae_model is not None:
            print(f"Successfully stored modified features.")
            # If I want to specify the shape I could f.e. iterate: for name in self.layer_names: self.activations[name].shape}
        else:
            print(f"Successfully stored original features.")