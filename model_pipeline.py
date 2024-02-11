import torch 

from utils import *
from evaluate_feature_maps import *
from get_sae_image_size import GetSaeImgSize

class ModelPipeline:
    '''
    - training the original model
    - perfoming inference through the original model
    - training the SAE
    - performing inference through the modified model (original model + SAE)
    - storing activations
    - returning statistics, such as losses, sparsity, accuracy, polysemanticity,...

    Explanations of selected parameters
    ----------
    use_sae: bool
        If True, the output of the specified layer of the original model is modified by passing it through an SAE
    train_sae: bool
        If True, the SAE is trained. This parameter can only be True if use_sae is True
    train_original_model: bool
        If True, the original model is trained. This parameter can only be True if use_sae is False
    Note: there is no use_model parameter, since the original model is always used

    store_activations: bool
        If True, the activations of all layers of the original model are stored (for the whole dataset, for the last epoch),
        and those of the modified model if it was used, and the encouder output of the SAE.
    
    prof: torch.profiler.profile
        If not None, the profiler is used to profile the forward pass of the model
    '''
    # constructor of the class (__init__ method)
    def __init__(self, 
                 device,
                 train_dataloader,
                 category_names,
                 layer_names, 
                 activation_threshold,
                 prof=None,
                 use_sae=None,
                 train_sae=None, 
                 train_original_model=None,
                 store_activations=None,
                 compute_feature_similarity=None,
                 activations_folder_path=None, 
                 sae_weights_folder_path=None,
                 model_weights_folder_path=None,
                 evaluation_results_folder_path=None): 
        self.device = device
        self.train_dataloader = train_dataloader
        self.category_names = category_names
        self.layer_names = layer_names
        self.activation_threshold = activation_threshold
        self.prof = prof 
        self.store_activations = store_activations
        self.compute_feature_similarity = compute_feature_similarity

        # Boolean parameters to control the behavior of the class
        self.use_sae = use_sae
        self.train_sae = train_sae
        self.train_original_model = train_original_model
        if self.train_sae and not self.use_sae:
            raise ValueError("If you want to train the SAE, you need to set use_sae=True.")
        if self.use_sae and self.train_original_model:
            raise ValueError("You can only train the original model, when use_sae=False.")

        # Folder paths
        self.activations_folder_path = activations_folder_path
        self.sae_weights_folder_path = sae_weights_folder_path
        self.model_weights_folder_path = model_weights_folder_path
        self.evaluation_results_folder_path = evaluation_results_folder_path

        # Compute basic dataset statistics
        self.num_samples = len(self.train_dataloader.dataset) # alternatively: train_dataset.__len__()
        self.num_batches = len(self.train_dataloader)
        self.num_classes = len(self.train_dataloader.dataset.classes) # alternatively: len(category_names)
        
        self.hooks = [] # list to store the hooks, 
        # so that we can remove them again later for using the model without hooks


    def instantiate_models(self, 
                           model_name, 
                           img_size, 
                           model_optimizer_name=None,
                           model_criterion_name=None,
                           model_learning_rate=None,
                           model_params=None,
                           sae_model_name=None,
                           sae_expansion_factor=None,
                           sae_lambda_sparse=None,
                           sae_optimizer_name=None,
                           sae_criterion_name=None,
                           sae_learning_rate=None,
                           sae_params=None,
                           sae_params_1=None):
        self.sae_params = sae_params
        self.sae_params_1 = sae_params_1
        self.model_params = model_params
        self.sae_expansion_factor = sae_expansion_factor
        self.sae_lambda_sparse = sae_lambda_sparse
        self.model_criterion = get_criterion(model_criterion_name)

        if self.train_original_model:
            self.model = load_model(model_name, img_size)
            self.model = self.model.to(self.device)
            self.model.train()
            self.model_optimizer = get_optimizer(model_optimizer_name, self.model, model_learning_rate)
        else:
            self.model = load_pretrained_model(model_name,
                                                img_size,
                                                self.model_weights_folder_path,
                                                params=self.model_params)
            self.model = self.model.to(self.device)
            # We set the original model to eval mode (changes the behavior of certain layers, such as dropout)
            self.model.eval()
            # We freeze the model by disabling gradients
            for param in self.model.parameters():
                param.requires_grad = False

        if self.use_sae:
            sae_img_size_getter = GetSaeImgSize(self.model, self.layer_names, self.train_dataloader)
            self.sae_img_size = sae_img_size_getter.get_sae_img_size()
            self.sae_criterion = get_criterion(sae_criterion_name, sae_lambda_sparse)

            if self.train_sae:
                self.sae_model = load_model(sae_model_name, self.sae_img_size, sae_expansion_factor)
                self.sae_model = self.sae_model.to(self.device)
                self.sae_model.train()
                self.sae_optimizer = get_optimizer(sae_optimizer_name, self.sae_model, sae_learning_rate)
            else:
                # if we only perform inference we load a pretrained SAE model
                self.sae_model = load_pretrained_model(sae_model_name,
                                                self.sae_img_size,
                                                self.sae_weights_folder_path,
                                                sae_expansion_factor=sae_expansion_factor,
                                                layer_names=self.layer_names,
                                                params=self.sae_params)
                self.sae_model = self.sae_model.to(self.device)
                self.sae_model.eval()
                for param in self.sae_model.parameters():
                    param.requires_grad = False

            # we also create a copy of the original model so that we have 2 models:
            # one adjusted model + one original model --> enables us to compare the outputs of those models
            self.model_copy = load_pretrained_model(model_name,
                                            img_size,
                                            self.model_weights_folder_path,
                                            params=self.model_params)
            self.model_copy = self.model_copy.to(self.device)
            self.model_copy.eval()
            for param in self.model_copy.parameters():
                param.requires_grad = False

    def store_activations_sparsity_polysemanticity(self, model_key, output, name):
        '''
        This function stores activations (if desired), sparsity info and polysemanticity level.

        Parameters
        ----------
        model_key : str
            "original" if not use_sae (layer outputs of original model)
            "modified" if use_sae (layer outputs of modified model)
            "sae" (encoder outputs of SAE)
        output : torch.Tensor
        name : str
            The name of the current layer of the model
        '''
        # store the activations of the current layer
        if self.compute_feature_similarity or self.store_activations:
            if (name,model_key) not in self.activations:
                self.activations[(name,model_key)] = []
            self.activations[(name,model_key)].append(output)

        # store the sparsity info of current layer
        activated_units, total_units = measure_activating_units(output, self.activation_threshold)
        if (name,model_key) not in self.sparsity:
            self.sparsity[(name, model_key)] = (activated_units, total_units)
        else:
            self.sparsity[(name, model_key)] = (self.sparsity[(name,model_key)][0] + activated_units, self.sparsity[(name,model_key)][1] + total_units)

        # store the relative sparsity of current layer
        if model_key == 'sae':
            # if we are dealing with the encoder output of the SAE, we scale the sparsity by the expansion factor
            factor = self.sae_expansion_factor
        else:
            factor = 1
        if (name,model_key) not in self.relative_sparsity:
            self.relative_sparsity[(name,model_key)] = compute_sparsity(activated_units, total_units, len(self.train_dataloader), factor)
        else:
            self.relative_sparsity[(name,model_key)] = self.relative_sparsity[(name,model_key)] + compute_sparsity(activated_units, total_units, len(self.train_dataloader), factor)

        # store the polysemanticity level of current layer
        mean_active_classes_per_neuron, std_active_classes_per_neuron = polysemanticity_level(output, self.targets, self.num_classes, self.activation_threshold)
        if (name,model_key) not in self.polysemanticity:
            self.polysemanticity[(name,model_key)] = (mean_active_classes_per_neuron, std_active_classes_per_neuron)
        else:
            self.polysemanticity[(name,model_key)] = (self.polysemanticity[(name,model_key)][0] + mean_active_classes_per_neuron, self.polysemanticity[(name,model_key)][1] + std_active_classes_per_neuron)


    def hook(self, module, input, output, name, use_sae, train_sae):
        '''
        Retrieve and possibly modify outputs of the original model
        Shape of variable output: [channels, height, width] --> no batch dimension, since we iterate over each batch
        '''                   
        # we store quantities of the original model
        if not use_sae:
            self.store_activations_sparsity_polysemanticity(model_key='original', output=output, name=name)
        
        # use the sae to modify the output of the specified layer of the original model
        if use_sae and name in self.layer_names:
            # the output of the specified layer of the original model is the input of the SAE            
            encoder_output, decoder_output = self.sae_model(output) 
            rec_loss, l1_loss = self.sae_criterion(encoder_output, decoder_output, output) # the inputs are the targets
            loss = rec_loss + self.sae_lambda_sparse*l1_loss
            self.sae_rec_loss += rec_loss.item()
            self.sae_l1_loss += l1_loss.item()
            self.sae_loss += loss.item()

            if train_sae:
                self.sae_optimizer.zero_grad()
                loss.backward()
                self.sae_optimizer.step()
            
            # store quantities of the encoder output
            self.store_activations_sparsity_polysemanticity(model_key='sae', output=encoder_output, name=name)

            # we pass the decoder_output back to the original model
            output = decoder_output

        # we store quantities of the modified model, in case we passed the layer output through the SAE, 
        # then we store quantities of the sae decoder output here
        if use_sae:
            self.store_activations_sparsity_polysemanticity(model_key='modified', output=output, name=name)
        
        return output
    
    def hook_2(self, module, input, output, name):
        '''
        This hook will only be used when use_sae=True to be registered for the original model, which is
        evaluated in parallel to the modified model (which is either evaluated/inference or trained). 
        The hook extracts the intermediate activations from the original model, allowing to compute the 
        feature similarity to the modified model.
        '''
        self.store_activations_sparsity_polysemanticity(model_key='original', output=output, name=name)


    def register_hooks(self, use_sae, train_sae):
        module_names = get_module_names(self.model)
        for name in module_names:
            m = getattr(self.model, name)
            hook = m.register_forward_hook(lambda module, inp, out, name=name, use_sae=use_sae, train_sae=train_sae: self.hook(module, inp, out, name, use_sae, train_sae))
            self.hooks.append(hook)
        # The below line works successfully for ResNet50:
        #self.model.layer1[0].conv3.register_forward_hook(lambda module, inp, out, name='model.layer1[0].conv3': self.hook(module, inp, out, name))
        # if I do this in ResNet50: m = getattr(module, 'layer1[0].conv3') --> not an attribute of the model
        
        if use_sae: # see method description of hook_2 for an explanation on what this is doing
            for name in module_names:
                m = getattr(self.model_copy, name)
                m.register_forward_hook(lambda module, inp, out, name=name: self.hook_2(module, inp, out, name))
          
    def epoch_forward_pass(self, use_sae, train_sae, train_original_model):
        # Once we perform the forward pass, the hook will store the activations 
        # (and modify the output of the specified layer if desired)
        # As we iterate over batches, the activations will be appended to the dictionary
        batch_idx = 0
        self.register_hooks(use_sae, train_sae) # registering the hook within the for loop will lead to undesired behavior
        # as the hook will be registered multiple times --> activations will be captured multiple times!

        # Placeholders for storing values, reset for each epoch
        self.activations = {} 
        self.sparsity = {} 
        self.polysemanticity = {} 
        self.relative_sparsity = {}
        self.accuracy = 0.0
        self.model_loss = 0.0
        
        if use_sae:
            self.sae_loss = 0.0
            self.sae_rec_loss = 0.0
            self.sae_l1_loss = 0.0
            self.kld = 0.0
            self.perc_same_classification = 0.0
            self.activation_similarity = {}

        for batch in self.train_dataloader:
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                inputs, targets = batch
            else:
                raise ValueError("Unexpected data format from dataloader")
                        
            inputs, self.targets = inputs.to(self.device), targets.to(self.device)
            outputs = self.model(inputs)
            loss = self.model_criterion(outputs, self.targets)
            self.model_loss += loss.item()

            # compute accuracy
            _, class_ids = torch.max(outputs, 1)
            self.accuracy += torch.sum(class_ids == targets).item() / targets.size(0)

            if use_sae:
                # We use (a copy of) the original model to get its outputs and activations
                # and then compare them with those of the modified model
                # hook_2 should be registered on self.model_copy to get the activations
                outputs_original = self.model_copy(inputs)

                # we apply first softmax (--> prob. distr.) and then log
                log_prob_original = F.log_softmax(outputs_original, dim=1)
                log_prob_modified = F.log_softmax(outputs, dim=1)
                # log_target = True means that we pass log(target) instead of target (second argument of kl_div)
                kld = F.kl_div(log_prob_original, log_prob_modified, reduction='sum', log_target=True)
                # see the usage example in https://pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html
                self.kld += kld.item()

                # we calculate the percentage of same classifications of modified
                # and original model
                _, class_ids_original = torch.max(outputs_original, dim=1)
                self.perc_same_classification += (class_ids_original == class_ids).sum().item() / class_ids_original.size(0)
        
            if train_original_model:
                self.model_optimizer.zero_grad()
                loss.backward()
                self.model_optimizer.step()

            batch_idx += 1
            # do a profiler step if a profiler is provided
            if self.prof is not None:
                self.prof.step()

        # The quantities which we simply added together, we normalize by the number of samples; 
        # the quantities which we added together and then divided by the number of samples for each batch, we normalize by the number of batches
        self.sparsity = {k: (v[0]/self.num_samples, v[1]/self.num_samples) for k, v in self.sparsity.items()}
        self.polysemanticity = {k: (v[0]/self.num_samples, v[1]/self.num_samples) for k, v in self.polysemanticity.items()}
        self.relative_sparsity = {k: v/self.num_batches for k, v in self.relative_sparsity.items()}
        self.accuracy = self.accuracy/self.num_batches
      
        if use_sae:
            self.kld = self.kld/self.num_samples
            self.perc_same_classification = self.perc_same_classification/self.num_batches

            # we calculate the feature similarity between the modified and original model
            module_names = get_module_names(self.model)
            for name_1 in module_names:
                activations = [(k,v) for k, v in self.activations.items() if k[0] == name_1]
                # activations should be of the form (with len(activations)=3)
                # [((name_1, 'modified'), [list of tensors, one for each batch]), 
                #  ((name_1, 'original'), [list of tensors, one for each batch])]
                # and if we inserted an SAE at the given layer with name_1, then 
                # ((name_1, 'sae'), [list of tensors, one for each batch]) is the first entry of
                # activations
                # we check whether activations has the expected shape
                if (activations[-2][0][1] == "modified" and activations[-1][0][1] == "original"):
                    activation_list_1 = activations[-1][1]
                    activation_list_2 = activations[-2][1]
                    dist_mean = 0.0
                    dist_std = 0.0
                    for act1, act2 in zip(activation_list_1, activation_list_2):
                        activation_1 = act1.to(self.device)
                        activation_2 = act2.to(self.device)
                        # dist is the distance between each pair of samples --> dimension is [batch_size]
                        sample_dist = torch.linalg.norm(activation_1 - activation_2, dim=1)
                        dist_mean += sample_dist.mean().item()
                        dist_std += sample_dist.std().item()   
                    # We get the mean mean over all batches
                    dist_mean = dist_mean/self.num_batches
                    # We get the mean std over all batches
                    dist_std = dist_std/self.num_batches
                    self.activation_similarity[name_1] = (dist_mean, dist_std)
                else:
                    raise ValueError("Activations has the wrong shape for evaluating feature similarity.")


    def deploy_model(self, num_epochs, wandb_status):
        if self.use_sae and self.train_sae:
            print("Starting SAE training...")
            num_epochs += 1 # since we use the first epoch for inference to calculate 
            # the initial loss (and other values), and then num_epochs of training
        elif self.use_sae and not self.train_sae:
            print("Starting inference through the modified model...")
        elif self.train_original_model:
            print("Starting training of the original model...")
            num_epochs += 1
        else:
            print("Starting inference through the original model...")

        for epoch in range(num_epochs):
            if (self.train_sae and epoch==0) or (self.use_sae and not self.train_sae):
                use_sae=True
                train_sae=False
                train_original_model=False
            elif (self.train_sae and epoch>0):
                use_sae=True
                train_sae=True
                train_original_model=False
            elif (not self.use_sae and self.train_original_model and epoch==0) or (not self.use_sae and not self.train_original_model):
                use_sae=False
                train_sae=False
                train_original_model=False
            elif (not self.use_sae and self.train_original_model and epoch>0):
                use_sae=False
                train_sae=False
                train_original_model=True
            else: 
                raise ValueError("Parameters are not set correctly.")

            self.epoch_forward_pass(use_sae=use_sae, train_sae=train_sae, train_original_model=train_original_model)
            # the above method has no return values but it updates the class attributes, such as self.activations
            
            print("---------------------------")
            print(f"Epoch {epoch+1}/{num_epochs} | Model loss: {self.model_loss:.4f} | Model accuracy: {self.accuracy:.4f}")
            if wandb_status:
                wandb.log({"Model loss": self.model_loss, "model accuracy": self.accuracy}, step=epoch, commit=False)
            for name in self.sparsity.keys(): 
            # the names are the same for the polysemanticity and relative sparsity dictionaries
            # hence, it suffices to iterate over the keys of the sparsity dictionary
                if name[1] == "sae":
                    entity = "encoder output"
                elif name[1] == "original":
                    entity = "original model"
                elif name[1] == "modified":
                    entity = "modified model"
                # for now, only print/log results for the specified layer and the last layer, which is 'fc3'
                if name[0] in self.layer_names or name[0] == 'fc3':
                    print(f"Activated/total units; {entity}, layer {name[0]}: {self.sparsity[name][0]:.4f} | {self.sparsity[name][1]:.4f}")
                    print(f"Relative sparsity; {entity}, layer {name[0]}: {self.relative_sparsity[name]:.4f}")
                    print(f"Mean and std of active classes per neuron; {entity}, layer {name[0]}: {self.polysemanticity[name][0]:.4f} | {self.polysemanticity[name][1]:.4f}")
                    if wandb_status:
                        wandb.log({f"Relative sparsity; {entity}, layer {name[0]}": self.relative_sparsity[name]}, step=epoch, commit=False)
                        wandb.log({f"Mean active classes per neuron; {entity}, layer {name[0]}": self.polysemanticity[name][0]}, step=epoch, commit=False)
                        wandb.log({f"Std active classes per neuron; {entity}, layer {name[0]}": self.polysemanticity[name][1]}, step=epoch, commit=False)
            if self.use_sae:
                print(f"SAE loss: {self.sae_loss:.4f} | SAE rec. loss: {self.sae_rec_loss:.4f} | SAE l1 loss: {self.sae_l1_loss:.4f} | KLD: {self.kld:.4f} | Perc same classifications: {self.perc_same_classification:.4f}")
                print(f"Mean and std of feature similarity (L2 loss) between modified and original model; {entity}, layer {name[0]}: {self.activation_similarity[name[0]][0]:.4f} | {self.activation_similarity[name[0]][1]:.4f}")
                if wandb_status:
                    wandb.log({"SAE loss": self.sae_loss, "SAE rec. loss": self.sae_rec_loss, "SAE l1 loss": self.sae_l1_loss, "KLD": self.kld, "Perc same classifications": self.perc_same_classification}, step=epoch, commit=False)
                    wandb.log({f"Mean feature similarity (L2 loss) between modified and original model; {entity}, layer {name[0]}": self.activation_similarity[name][0]}, step=epoch, commit=False)
                    wandb.log({f"Std feature similarity (L2 loss) between modified and original model; {entity}, layer {name[0]}": self.activation_similarity[name][1]}, step=epoch, commit=False)
            if wandb_status:
                wandb.log({}, commit=True) # commit the above logs
        
        print("---------------------------")
        if self.use_sae:
            if self.train_sae:
                print("SAE training completed.")
                # store SAE model weights
                save_model_weights(self.sae_model, self.sae_weights_folder_path, layer_names=self.layer_names, params=self.sae_params)
            else:  
                print("Inference through the modified model completed.")
            # We store the sae_rec_loss and sae_l1_loss from the last epoch
            store_sae_losses(self.evaluation_results_folder_path, self.layer_names, self.sae_params_1, self.sae_lambda_sparse, self.sae_expansion_factor, self.sae_rec_loss, self.sae_l1_loss)
        elif self.train_original_model:
            print("Training of the original model completed.")
            # store original model weights
            save_model_weights(self.model, self.model_weights_folder_path, params=self.model_params)
        else:
            print("Inference through the original model completed.")

        # We display some sample input images with their corresponding true and predicted labels as a sanity check 
        # We remove the hooks as we don't need them anymore; using the model with hooks for the below visualization
        # will return some error
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        params = {**self.model_params, **self.sae_params} # merge model_params and sae_params
        if wandb_status:
            log_image_table(self.train_dataloader,
                            self.category_names,
                            model=self.model, 
                            device=self.device)
        else:
            show_classification_with_images(self.train_dataloader,
                                            self.category_names,
                                            folder_path=self.evaluation_results_folder_path,
                                            model=self.model, 
                                            device=self.device,
                                            params=params)
        
        # store the feature maps (of up to those 3 types: of original model, modified model and encoder output)
        # after the last epoch 
        if self.store_activations:
            self.activations = {k: torch.cat(v, dim=0) for k, v in self.activations.items()}
            store_feature_maps(self.activations, self.activations_folder_path, params=params)
            print(f"Successfully stored activations.")
            # Previous code to store activations:
            #for name in self.activations.keys():
            #    self.activations[name] = torch.cat(self.activations[name], dim=0)
            #store_feature_maps(self.activations, self.activations_folder_path, params=params)