import torch 
from tqdm import tqdm

from utils import *
from get_sae_image_size import GetSaeImgSize

class ModelPipeline:
    '''
    This class is used to perform the following tasks:
    - training the original model
    - perfoming inference through the original model
    - training the SAE
    - performing inference through the modified model (original model + SAE)
    - storing activations
    - returning statistics, such as losses, sparsity, accuracy, ...

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
        If not None, the profiler is used to profile the forward pass of the model to identify inefficiencies in the code.
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
        This function stores activations (if desired), sparsity info and polysemanticity 
        level for one batch of data.

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
        # For measuring whether a certain neuron is active or not, we consider the absolute value of the activation. 
        # For instance, if the threshold is 0.1, then an activation of -0.4 is also considered as active. 
        absolute_output = output.abs()

        # store the activations of the current layer
        if self.compute_feature_similarity or self.store_activations:
            if (name,model_key) not in self.activations:
                self.activations[(name,model_key)] = []
            self.activations[(name,model_key)].append(output)

        # store the sparsity info of current layer
        inactive_neurons, number_active_neurons, number_total_neurons = measure_activating_neurons(absolute_output, self.activation_threshold)
        if (name,model_key) not in self.number_active_neurons:
            self.number_active_neurons[(name, model_key)] = (number_active_neurons, number_total_neurons)
        else:
            self.number_active_neurons[(name, model_key)] = (self.number_active_neurons[(name,model_key)][0] + number_active_neurons, 
                                                             self.number_active_neurons[(name,model_key)][1]) 
            # the total number of neurons is the same for all samples, hence we don't need to sum it up

        if (name,model_key) not in self.dead_neurons:
            self.dead_neurons[(name, model_key)] = inactive_neurons
        else:
            # dead_neurons is of the form [True,False,False,True,...] with size of the respective layer, where "True" stands for dead
            # neuron and "False" stands for "active" neuron. We have the previous dead_neurons entry and a new one. A neuron 
            # is counted as dead if it was dead before and is still dead, otherwise it is counted as "active". 
            # This can be achieved through pointwise multiplication.
            self.dead_neurons[(name, model_key)] = self.dead_neurons[(name,model_key)] * inactive_neurons

        # store a matrix of size [#neurons, #classes] with one entry for each neuron (of the current layer) and each class, with the number of how often this neuron 
        # is active on samples from that class for the current batch; measure of polysemanticity
        number_active_classes_per_neuron = active_classes_per_neuron_aux(absolute_output, self.targets, self.num_classes, self.activation_threshold)
        if (name,model_key) not in self.number_active_classes_per_neuron:
            self.number_active_classes_per_neuron[(name,model_key)] = number_active_classes_per_neuron
        else:
            self.number_active_classes_per_neuron[(name,model_key)] = self.number_active_classes_per_neuron[(name,model_key)] + number_active_classes_per_neuron


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
            if use_sae: # see method description of hook_2 for an explanation on what this is doing
                m1 = getattr(self.model_copy, name)
                hook1 = m1.register_forward_hook(lambda module, inp, out, name=name: self.hook_2(module, inp, out, name))
                self.hooks.append(hook1)

        # The below line works successfully for ResNet50:
        #self.model.layer1[0].conv3.register_forward_hook(lambda module, inp, out, name='model.layer1[0].conv3': self.hook(module, inp, out, name))
        # if I do this in ResNet50: m = getattr(module, 'layer1[0].conv3') --> not an attribute of the model
                        
    def epoch_forward_pass(self, use_sae, train_sae, train_original_model):
        # Once we perform the forward pass, the hook will store the activations 
        # (and modify the output of the specified layer if desired)
        # As we iterate over batches, the activations will be appended to the dictionary
        self.batch_idx = 0
        self.register_hooks(use_sae, train_sae) # registering the hook within the for loop will lead to undesired behavior
        # as the hook will be registered multiple times --> activations will be captured multiple times!

        # Placeholders for storing values, reset for each epoch
        self.activations = {} 
        self.number_active_classes_per_neuron = {} 
        self.number_active_neurons = {}
        self.accuracy = 0.0
        self.model_loss = 0.0
        
        if use_sae:
            self.sae_loss = 0.0
            self.sae_rec_loss = 0.0
            self.sae_l1_loss = 0.0
            self.kld = 0.0
            self.perc_same_classification = 0.0
            self.activation_similarity = {}

        for batch in tqdm(self.train_dataloader):
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                inputs, targets = batch
            else:
                raise ValueError("Unexpected data format from dataloader")
            
            self.epoch_progress_bar.update(1)
                        
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

            self.batch_idx += 1
            # do a profiler step if a profiler is provided
            if self.prof is not None:
                self.prof.step()

        # The quantities which we simply added together, we normalize by the number of samples; 
        # the quantities which we added together and then divided by the number of samples for each batch, we normalize by the number of batches
        self.number_active_neurons = {k: (v[0]/self.num_samples, v[1]) for k, v in self.number_active_neurons.items()}
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

                    # we check whether the length of both activation lists corresponds to the number of batches
                    if len(activation_list_1) != self.num_batches or len(activation_list_2) != self.num_batches:
                        raise ValueError(f"For layer {name_1}: The length of the activation lists for computing feature similarity (length of activation list of modified model {len(activation_list_2)}, original model {len(activation_list_1)}) does not correspond to the number of batches {self.num_batches}.")

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

        # We remove the hooks after every epoch. Otherwise, we will have 2 hooks for the next epoch.
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def deploy_model(self, num_epochs, dead_neurons_epochs, wandb_status):
        dead_neurons_epochs_counter = 0
        self.dead_neurons = {}

        # if we are evaluating the modified model or the original model, we only perform one epoch
        if self.use_sae and not self.train_sae:
            num_epochs = 1
        elif not self.use_sae and not self.train_original_model:
            num_epochs = 1

        for epoch in range(num_epochs):
            if (self.train_sae and epoch==0) or (self.use_sae and not self.train_sae):
                use_sae=True
                train_sae=False
                train_original_model=False
                if self.train_sae and epoch==0:
                    print("Performing one inference pass before training of the SAE...")
                elif self.use_sae and not self.train_sae and epoch==0:
                    print("Starting inference through the modified model...")
            elif (self.train_sae and epoch>0):
                print("Starting SAE training...")
                use_sae=True
                train_sae=True
                train_original_model=False
            elif (not self.use_sae and self.train_original_model and epoch==0) or (not self.use_sae and not self.train_original_model):
                use_sae=False
                train_sae=False
                train_original_model=False
                if not self.use_sae and self.train_original_model and epoch==0:
                    print("Performing one inference pass before training of the original model...")
                elif not self.use_sae and not self.train_original_model and epoch==0:
                    print("Starting inference through the original model...")
            elif (not self.use_sae and self.train_original_model and epoch>0):
                print("Starting training of the original model...")
                use_sae=False
                train_sae=False
                train_original_model=True
            else: 
                raise ValueError("Parameters are not set correctly.")
            
            # initialize the progress bar for showing progress of the current epoch
            self.epoch_progress_bar = tqdm(total=num_epochs, desc=f'Epoch {epoch}')
            
            if dead_neurons_epochs_counter == dead_neurons_epochs:
                self.dead_neurons = {}
            elif dead_neurons_epochs_counter == 2*dead_neurons_epochs:
                # re-initialize weights of dead neurons in the SAE (recall that the base model 
                # is assumed to be given and frozen). In particular, HOW IS IT DONE??? CHECK DICTIONARY LEARNIGN PAPER 
                # BY ANTHROPIC
                # Then, we let those neurons participate
                # in training for dead_neurons_epochs
                # TO DO !!!!!!!
                dead_neurons_epochs_counter = 0
                self.dead_neurons = {}
            dead_neurons_epochs_counter += 1

            self.epoch_forward_pass(use_sae=use_sae, 
                                    train_sae=train_sae, 
                                    train_original_model=train_original_model)
            # the above method has no return values but it updates the class attributes, such as self.activations
            self.epoch_progress_bar.close()

            # count the number of dead neurons and report them (if number dead neurons is not None)
            # for each key in self.dead_neurons, count the number of "True"'s
            number_dead_neurons = compute_number_dead_neurons(self.dead_neurons)

            print("---------------------------")
            print(f"Epoch {epoch}/{num_epochs} | Model loss: {self.model_loss:.4f} | Model accuracy: {self.accuracy:.4f}")
            if self.use_sae:
                print(f"SAE loss: {self.sae_loss:.4f} | SAE rec. loss: {self.sae_rec_loss:.4f} | SAE l1 loss: {self.sae_l1_loss:.4f} | KLD: {self.kld:.4f} | Perc same classifications: {self.perc_same_classification:.4f}")
            if wandb_status:
                wandb.log({"Model loss": self.model_loss, "model accuracy": self.accuracy}, step=epoch, commit=False)
                if self.use_sae:
                    wandb.log({"SAE loss": self.sae_loss, "SAE rec. loss": self.sae_rec_loss, "SAE l1 loss": self.sae_l1_loss, "KLD": self.kld, "Perc same classifications": self.perc_same_classification}, step=epoch, commit=False)
                
            mean_active_classes_per_neuron = {}
            sparsity_dict = {}

            # We show per model layer evaluation metrics and compute some quantities 
            for name in self.number_active_neurons.keys(): 
            # the names are the same for the polysemanticity and relative sparsity dictionaries
            # hence, it suffices to iterate over the keys of the sparsity dictionary
                model_key = name[1]
                if model_key == "original":
                    entity = "original model"
                elif model_key == "sae":
                    entity = "encoder output"
                elif model_key == "modified":
                    entity = "modified model"

                # Compute relative sparsity
                # Notice that we compute the number of dead neurons at the end of an epoch (here), once it's clear
                # which neurons are dead and which ones are used. On the other hand, the number of active neurons is 
                # summed up throughout the forward pass for each sample and then averaged (in the epoch_forward_pass method)
                                    
                # if we want to use the expansion factor
                #if name[1] == 'sae': # also used as model_key in other parts of the code
                #    factor = self.sae_expansion_factor
                #else:
                #    factor = 1
                average_activated_neurons = self.number_active_neurons[name][0]
                total_neurons = self.number_active_neurons[name][1] 
                # count the number of dead neurons, i.e., the occurences of "True" in the list
                number_used_neurons = total_neurons - number_dead_neurons[name]
                sparsity = compute_sparsity(average_activated_neurons, number_used_neurons)
                sparsity_dict[name] = sparsity
                
                # the following unit tests only work if dead neurons are measured over 1 epoch,
                # because number_active_classes_per_neuron is measured over 1 epoch and otherwise 
                # we couldnt compare them. Of course I could additionally measure active classes
                # over multiple epochs for comparison (by having another variable which stores those
                # values over several epochs)
                if dead_neurons_epochs == 1:
                    # ----------------------------------------------
                    # First we check that the number of rows of the number_active_classes_per_neuron matrix
                    # is equal to the number of neurons, i.e., to the length of dead_neurons
                    if self.number_active_classes_per_neuron[name].shape[0] != len(number_dead_neurons[name]):
                        raise ValueError(f"The number of rows of the number_active_classes_per_neuron matrix ({self.number_active_classes_per_neuron[name].shape[0]}) is not equal to the number of neurons ({len(number_dead_neurons[name])}).")

                    # Now we remove the rows/neurons from the matrix which correspond to dead neurons
                    a = self.number_active_classes_per_neuron[name][self.dead_neurons[name] == False]
                    # we check whether the number of rows of the matrix with dead neurons removed is equal to the number of used neurons
                    # this is equivalent to: the number of neurons which weren't active on any class is 
                    # equal to the number of dead neurons
                    if a.shape[0] != number_used_neurons:
                        raise ValueError(f"The number of rows of the number_active_classes_per_neuron matrix with dead neurons removed ({a.shape[0]}) is not equal to the number of used neurons ({number_used_neurons}).")

                    
                    mean_active_classes_per_neuron = a.mean()
                    std_active_classes_per_neuron = a.std()

                    # as a sanity check, the number of dead neurons should correspond to 
                    # the number of 0's in self.number_active_classes_per_neuron because a neuron is 
                    # dead iff it is active on 0 classes throughout this epoch
                    if number_dead_neurons[name] != (self.number_active_classes_per_neuron[name] == 0).sum().item():
                        raise ValueError(f"{name}: The number of dead neurons ({number_dead_neurons[name]}) is not equal to the number of neurons ({(self.number_active_classes_per_neuron[name] == 0).sum()}) which are active on 0 classes.")


                # number of active neurons should be less than or equal to the number of used neurons
                if average_activated_neurons > number_used_neurons:
                    raise ValueError(f"The number of active neurons ({average_activated_neurons}) is greater than the number of used neurons ({number_used_neurons}).")

                # for now, only print/log results for the specified layer and the last layer, which is 'fc3'
                if name[0] in self.layer_names or name[0] == 'fc3':
                    print("-------")
                    print(f"{entity}, Layer {name[0]}")
                    print(f"Activated/dead/total neurons: {average_activated_neurons:.2f} | {number_dead_neurons[name]} | {int(total_neurons)}")
                    print(f"Relative sparsity: {sparsity:.3f}")
                    print(f"Mean and std of active classes per neuron: {mean_active_classes_per_neuron:.4f} | {std_active_classes_per_neuron:.4f}")
                    if self.use_sae: 
                        print(f"Mean and std of feature similarity (L2 loss) between modified and original model: {self.activation_similarity[name[0]][0]:.4f} | {self.activation_similarity[name[0]][1]:.4f}")
                    if wandb_status:
                        # can I log a dictionary to wandb? --> yes, see https://docs.wandb.ai/guides/track/log
                        wandb.log({f"Activation_of_neurons/{entity}_layer_{name[0]}: Activated neurons": average_activated_neurons}, step=epoch, commit=False)
                        wandb.log({f"Activation_of_neurons/{entity}_layer_{name[0]}: Dead neurons": number_dead_neurons[name]}, step=epoch, commit=False)
                        wandb.log({f"Activation_of_neurons/{entity}_layer_{name[0]}: Total neurons": total_neurons}, step=epoch, commit=False)
                        wandb.log({f"Sparsity/{entity}_layer_{name[0]}": sparsity}, step=epoch, commit=False)
                        wandb.log({f"Active_classes_per_neuron/{entity}_layer_{name[0]}: Mean": mean_active_classes_per_neuron}, step=epoch, commit=False)
                        wandb.log({f"Active_classes_per_neuron/{entity}_layer_{name[0]}: Std": std_active_classes_per_neuron}, step=epoch, commit=False)
                        if self.use_sae:
                            wandb.log({f"Feature_similarity_L2loss_between_modified_and_original_model/{entity}_layer_{name[0]}: Mean": self.activation_similarity[name[0]][0]}, step=epoch, commit=False) 
                            wandb.log({f"Feature_similarity_L2loss_between_modified_and_original_model/{entity}_layer_{name[0]}: Std": self.activation_similarity[name[0]][1]}, step=epoch, commit=False) 
            if wandb_status:
                # wandb doesn't accept tuples as keys, so we convert them to strings
                number_dead_neurons = {f"Number_of_dead_neurons/{k[0]}_{k[1]}": v for k, v in number_dead_neurons.items()}
                #number_dead_neurons = {f"{k[0]}_{k[1]}": v for k, v in number_dead_neurons.items()}
                wandb.log(number_dead_neurons, step=epoch, commit=False) # overview of number of dead neurons for all layers
                wandb.log({}, commit=True) # commit the above logs
            
            # printing statistics for all layers
            print("-------")
            print("Dead neurons: ", number_dead_neurons)
        print("---------------------------")
        
        if self.use_sae:
            if self.train_sae:
                print("SAE training completed.")
                # store SAE model weights
                save_model_weights(self.sae_model, self.sae_weights_folder_path, layer_names=self.layer_names, params=self.sae_params)
            else:  
                print("Inference through the modified model completed.")
            # We store the sae_rec_loss and sae_l1_loss from the last epoch
            store_sae_eval_results(self.evaluation_results_folder_path, self.layer_names, self.sae_params_1, self.sae_lambda_sparse, self.sae_expansion_factor, self.sae_rec_loss, self.sae_l1_loss, sparsity_dict)
        elif self.train_original_model:
            print("Training of the original model completed.")
            # store original model weights
            save_model_weights(self.model, self.model_weights_folder_path, params=self.model_params)
        else:
            print("Inference through the original model completed.")

        # We display some sample input images with their corresponding true and predicted labels as a sanity check 
        # Apart from that we display the distribution of the number of classes that each neuron is active on, for the last epoch
        params = {**self.model_params, **self.sae_params} # merge model_params and sae_params

        plot_active_classes_per_neuron(self.number_active_classes_per_neuron, 
                                       self.layer_names,
                                       num_classes=self.num_classes,
                                       folder_path=self.evaluation_results_folder_path, 
                                       params=params, 
                                       wandb_status=wandb_status)
        # not sure yet if I should just store the plot to wandb directly or rather as a wandb table as below?

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