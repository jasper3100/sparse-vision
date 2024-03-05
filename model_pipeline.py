import torch 
from tqdm import tqdm
from torchvision import transforms

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

    prof: torch.profiler.profile
        If not None, the profiler is used to profile the forward pass of the model to identify inefficiencies in the code.
    '''
    # constructor of the class (__init__ method)
    def __init__(self, 
                 device,
                 train_dataloader,
                 val_dataloader,
                 category_names,
                 layer_names, 
                 activation_threshold,
                 wandb_status,
                 prof=None,
                 use_sae=None,
                 train_sae=None, 
                 train_original_model=None,
                 sae_weights_folder_path=None,
                 model_weights_folder_path=None,
                 evaluation_results_folder_path=None,
                 dead_neurons_steps=None,
                 sae_batch_size=None): 
        self.device = device
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.category_names = category_names
        self.layer_names = layer_names
        self.activation_threshold = activation_threshold
        self.wandb_status = wandb_status
        self.prof = prof 
        self.dead_neurons_steps = dead_neurons_steps
        self.sae_batch_size = sae_batch_size

        # Boolean parameters to control the behavior of the class
        self.use_sae = use_sae
        self.train_sae = train_sae
        self.train_original_model = train_original_model
        if self.train_sae and not self.use_sae:
            raise ValueError("If you want to train the SAE, you need to set use_sae=True.")
        if self.use_sae and self.train_original_model:
            raise ValueError("You can only train the original model, when use_sae=False.")

        # Folder paths
        self.sae_weights_folder_path = sae_weights_folder_path
        self.model_weights_folder_path = model_weights_folder_path
        self.evaluation_results_folder_path = evaluation_results_folder_path

        # Compute basic dataset statistics
        self.num_train_samples = len(train_dataloader.dataset) # alternatively: train_dataset.__len__()
        #self.num_train_batches = len(train_dataloader)
        self.num_classes = len(train_dataloader.dataset.classes) # alternatively: len(category_names)
        #self.num_eval_batches = len(val_dataloader)

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
        self.img_size = img_size
        self.sae_params = sae_params
        self.sae_params_1 = sae_params_1
        self.model_params = model_params
        self.sae_expansion_factor = sae_expansion_factor
        self.sae_lambda_sparse = sae_lambda_sparse
        self.model_criterion = get_criterion(model_criterion_name)

        if self.train_original_model:
            self.model = load_model(model_name, img_size)
            self.model = self.model.to(self.device)
            self.model_optimizer = get_optimizer(model_optimizer_name, self.model, model_learning_rate)
            # We don't specify whether the model is in training or evaluation mode here, because we do this 
            # in the epoch_forward_pass method where we first set it to train mode and then to eval mode to 
            # eval on the test set (this is done for every epoch)
        else:
            self.model = load_pretrained_model(model_name,
                                                img_size,
                                                self.model_weights_folder_path,
                                                params=self.model_params)
            self.model = self.model.to(self.device)
            # If we don't train the original model we only use it to perform inference,
            # hence we do the following: We set it to eval mode (changes the behavior of 
            # certain layers, such as dropout)
            self.model.eval()
            # and we freeze the model by disabling gradients
            for param in self.model.parameters():
                param.requires_grad = False

        if self.use_sae:
            sae_img_size_getter = GetSaeImgSize(self.model, self.layer_names, self.train_dataloader)
            self.sae_img_size = sae_img_size_getter.get_sae_img_size()
            self.sae_criterion = get_criterion(sae_criterion_name, sae_lambda_sparse, self.sae_batch_size)

            if self.train_sae:
                self.sae_model = load_model(sae_model_name, self.sae_img_size, sae_expansion_factor)
                self.sae_model = self.sae_model.to(self.device)
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

            # if we are using an SAE, we also create a copy of the original model so that we have 2 models
            # one modified model (with SAE) + one original model --> enables us to compare the outputs of those models
            # This model is always used in inference mode only
            self.model_copy = load_pretrained_model(model_name,
                                            img_size,
                                            self.model_weights_folder_path,
                                            params=self.model_params)
            self.model_copy = self.model_copy.to(self.device)
            self.model_copy.eval()
            for param in self.model_copy.parameters():
                param.requires_grad = False

    def compute_and_store_batch_wise_metrics(self, model_key, output, name):
        '''
        This computes and logs certain metrics for each batch.

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
        self.batch_activations[(name,model_key)] = output

        inactive_neurons, number_active_neurons, number_total_neurons = measure_activating_neurons(absolute_output, self.activation_threshold)
        self.batch_number_active_neurons[(name,model_key)] = (number_active_neurons, number_total_neurons)
        self.batch_dead_neurons[(name,model_key)] = inactive_neurons

        # store a matrix of size [#neurons, #classes] with one entry for each neuron (of the current layer) and each class, with the number of how often this neuron 
        # is active on samples from that class for the current batch; measure of polysemanticity
        active_classes_per_neuron = active_classes_per_neuron_aux(absolute_output, self.targets, self.num_classes, self.activation_threshold)
        self.batch_active_classes_per_neuron[(name,model_key)] = active_classes_per_neuron


    def hook(self, module, input, output, name, use_sae, train_sae):
        '''
        Retrieve and possibly modify outputs of the original model
        Shape of variable output: [channels, height, width] --> no batch dimension, since we iterate over each batch
        '''                   
        # we store quantities of the original model
        if not use_sae:
            self.compute_and_store_batch_wise_metrics(model_key='original', output=output, name=name)
        
        # use the sae to modify the output of the specified layer of the original model
        if use_sae and name in self.layer_names:
            # the output of the specified layer of the original model is the input of the SAE            
            encoder_output, decoder_output = self.sae_model(output) 
            rec_loss, l1_loss, nrmse_loss, rmse_loss = self.sae_criterion(encoder_output, decoder_output, output) # the inputs are the targets
            loss = rec_loss + self.sae_lambda_sparse*l1_loss
            self.batch_sae_rec_loss = rec_loss.item()
            self.batch_sae_l1_loss = l1_loss.item()
            self.batch_sae_loss = loss.item()
            self.batch_sae_nrmse_loss = nrmse_loss.item()
            self.batch_sae_rmse_loss = rmse_loss.item()

            if train_sae:
                self.sae_optimizer.zero_grad()
                loss.backward()
                self.sae_optimizer.step()
            
            # store quantities of the encoder output
            self.compute_and_store_batch_wise_metrics(model_key='sae', output=encoder_output, name=name)

            # we pass the decoder_output back to the original model
            output = decoder_output

        # we store quantities of the modified model, in case we passed the layer output through the SAE, 
        # then we store quantities of the sae decoder output here
        if use_sae:
            self.compute_and_store_batch_wise_metrics(model_key='modified', output=output, name=name)
        
        return output
    
    def hook_2(self, module, input, output, name):
        '''
        This hook will only be used when use_sae=True to be registered for the original model, which is
        evaluated in parallel to the modified model (which is either evaluated/inference or trained). 
        The hook extracts the intermediate activations from the original model, allowing to compute the 
        feature similarity to the modified model.
        '''
        self.compute_and_store_batch_wise_metrics(model_key='original', output=output, name=name)

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

    def epoch(self, train_or_eval, epoch, num_epochs):
        '''
        train_or_eval | self.use_sae | 
        "train"       | False        | train the original model
        "train"       | True         | train the SAE
        "eval"        | False        | evaluate the original model
        "eval"        | True         | evaluate the modified model        
        '''
        if train_or_eval == "train":
            dataloader = self.train_dataloader
            if not self.use_sae: # original model
                # set model to train mode and unfreeze parameters
                self.model.train()
                for param in self.model.parameters():
                    param.requires_grad = True
                train_sae = False
            else: # modified model
                # set model to eval mode and unfreeze parameters
                self.sae_model.train()
                for param in self.sae_model.parameters():
                    param.requires_grad = True
                train_sae = True

        elif train_or_eval == "eval":
            dataloader = self.val_dataloader
            if not self.use_sae: # original model
                # set model to eval mode and freeze parameters
                self.model.eval()
                for param in self.model.parameters():
                    param.requires_grad = False
                train_sae = False
            else: # modified model
                # set model to eval mode and freeze parameters
                self.sae_model.eval()
                for param in self.sae_model.parameters():
                    param.requires_grad = False
                train_sae = False
         
        self.register_hooks(self.use_sae, train_sae) # registering the hook within the batches for loop will lead to undesired behavior
        # as the hook will be registered multiple times --> activations will be captured multiple times!
        # we shouldn't use self.train_sae here, because even if self.train_sae==True, 
        # we evaluate the modified model after every training epoch 

        # During training, we perform data augmentation
        if train_or_eval == "train":
            # We define a list of data augmentations
            img_size_1 = self.img_size[-2:] 
            # for example, for cifar-10, img_size is (3, 32, 32) but we only need (32,32) here
            augmentations_list = [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                # after cropping a portion of the image, we resize it to the original size
                # because our model expects a certain input size. Moreover, torch.stack requires
                # that all images have the same size
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.0)),
                transforms.RandomResizedCrop(img_size_1, scale=(0.8, 1.0), ratio=(0.8, 1.2)) #, antialias=True)
            ]

        # if we are in eval mode, we keep track of quantities across batches
        if train_or_eval == "eval":
            accuracy = 0.0
            model_loss = 0.0
            if self.use_sae:
                sae_loss = 0.0
                sae_rec_loss = 0.0
                sae_l1_loss = 0.0
                sae_nrmse_loss = 0.0
                sae_rmse_loss = 0.0
                perc_same_classification = 0.0
                kld = 0.0
                activation_similarity = {}
                number_active_neurons = {}
                active_classes_per_neuron = {}
                eval_dead_neurons = {}

        ######## BATCH LOOP START ########
        with tqdm(dataloader, unit="batch") as tepoch:
            for batch in tepoch:
                tepoch.set_description(f'{train_or_eval} epoch {epoch}')

                self.batch_activations = {}
                self.batch_number_active_neurons = {}
                self.batch_dead_neurons = {}
                self.batch_active_classes_per_neuron = {}
                self.batch_activation_similarity = {}
                            
                if isinstance(batch, (list, tuple)) and len(batch) == 2:
                    inputs, targets = batch
                else:
                    raise ValueError("Unexpected data format from dataloader")
                
                inputs, self.targets = inputs.to(self.device), targets.to(self.device)
                if train_or_eval == "train":
                    # Randomly choose one augmentation from the list for each image in the batch
                    random_augmentation = transforms.RandomChoice(augmentations_list)
                    # Apply the randomly chosen augmentation to each image in the batch
                    inputs = torch.stack([random_augmentation(img) for img in inputs])
                    # the batch idx only counts batches used during training
                    self.batch_idx += 1
                outputs = self.model(inputs)
                batch_loss = self.model_criterion(outputs, self.targets)
                if train_or_eval == "train" and not self.use_sae: # original model
                    self.model_optimizer.zero_grad()
                    batch_loss.backward()
                    self.model_optimizer.step()
                batch_loss = batch_loss.item()
                _, class_ids = torch.max(outputs, 1)
                batch_accuracy = torch.sum(class_ids == targets).item() / targets.size(0)
                # if we are in eval mode we keep track of loss and accuracy across batches        
                if train_or_eval == "eval":
                    model_loss += batch_loss
                    accuracy += batch_accuracy

                ######## BATCH LOOP, USE SAE START ########
                if self.use_sae:
                    # We use (a copy of) the original model to get its outputs and activations
                    # and then compare them with those of the modified model
                    # hook_2 should be registered on self.model_copy to get the activations
                    outputs_original = self.model_copy(inputs)
                    # we apply first softmax (--> prob. distr.) and then log
                    log_prob_original = F.log_softmax(outputs_original, dim=1)
                    log_prob_modified = F.log_softmax(outputs, dim=1)
                    # log_target = True means that we pass log(target) instead of target (second argument of kl_div)
                    batch_kld = F.kl_div(log_prob_original, log_prob_modified, reduction='sum', log_target=True).item()
                    # see the usage example in https://pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html
                    batch_kld = batch_kld / inputs.size(0) # we divide by the batch size
                    # we calculate the percentage of same classifications of modified and original model
                    _, class_ids_original = torch.max(outputs_original, dim=1)
                    batch_perc_same_classification = (class_ids_original == class_ids).sum().item() / class_ids_original.size(0)

                    # we calculate the feature similarity between the modified and original model
                    self.batch_activation_similarity = feature_similarity(self.batch_activations,self.batch_activation_similarity,self.device)

                    self.batch_number_active_neurons = {k: (v[0]/inputs.size(0), v[1]) for k, v in self.batch_number_active_neurons.items()}
                    # inputs.size(0) is the batch size

                    # if we are in train mode we keep track of dead neurons possibly across several epochs
                    # depending on the value of dead_neurons_steps
                    if train_or_eval == "train":
                        for name in self.batch_dead_neurons.keys():
                            if name not in self.train_dead_neurons:
                                self.train_dead_neurons[name] = self.batch_dead_neurons[name]
                            else:
                                self.train_dead_neurons[name] = self.train_dead_neurons[name] * self.batch_dead_neurons[name]
                        # in train mode, we measure dead neurons for every batch
                        number_dead_neurons = compute_number_dead_neurons(self.train_dead_neurons)
                        _, sparsity_dict_1, number_active_classes_per_neuron, average_activated_neurons, total_neurons, mean_number_active_classes_per_neuron, std_number_active_classes_per_neuron = compute_sparsity(train_or_eval, 
                                                                                                self.sae_expansion_factor, 
                                                                                                self.batch_number_active_neurons, 
                                                                                                self.batch_active_classes_per_neuron,
                                                                                                num_classes=self.num_classes)
                        '''
                        During training, we also re-initialize dead neurons, which were dead over the last n steps (n=dead_neurons_steps)
                        Then, we let the model train with the new neurons for n steps
                        Then, we measure dead neurons for another n steps and re-initialize the dead neurons from those last n steps etc.
                        '''
                        if self.batch_idx % self.dead_neurons_steps == 0 and (self.batch_idx // self.dead_neurons_steps) % 2 == 1:
                            # first condition: self.batch_idx is a multiple of self.dead_neurons_steps
                            # second condition: self.batch_idx is an odd multiple of self.dead_neurons_steps
                            # --> self.batch_idx = n* self.dead_neurons_steps where n = 1,3,5,7,...

                            # we re-initialize dead neurons in the SAE (recall that the base model is assumed to be given 
                            # and frozen) by initializing the weights into a dead neuron (i.e. in the encoder)
                            # with a Kaiming uniform distribution and we set the corresponding bias to zero 
                            # this might be one of the most basic ways possible to re-initialize dead neurons

                            # get the dead neurons of the SAE encoder output
                            # (so far we assume that layer_names is a list with only one element)
                            dead_neurons_sae = self.train_dead_neurons[(self.layer_names[0], 'sae')]
                            self.sae_model.reset_encoder_weights(dead_neurons_sae=dead_neurons_sae, device=self.device, optimizer=self.sae_optimizer, batch_idx=self.batch_idx)

                            # we start measuring dead neurons from 0
                            self.train_dead_neurons = {}

                        elif self.batch_idx % self.dead_neurons_steps == 0 and (self.batch_idx // self.dead_neurons_steps) % 2 == 0 and self.batch_idx != 0:
                            # --> self.batch_idx = n* self.dead_neurons_steps where n = 2,4,6,8,... (the last condition excludes n=0)
                            self.train_dead_neurons = {}

                    # if we are in eval mode we accumulate the batch-wise quantities
                    if train_or_eval == "eval":
                        sae_rec_loss += self.batch_sae_rec_loss
                        sae_l1_loss += self.batch_sae_l1_loss
                        sae_loss += self.batch_sae_loss
                        sae_nrmse_loss += self.batch_sae_nrmse_loss
                        sae_rmse_loss += self.batch_sae_rmse_loss
                        kld += batch_kld
                        perc_same_classification += batch_perc_same_classification

                        for name in self.batch_number_active_neurons.keys():
                            if name not in number_active_neurons:
                                # the below dictionaries have the same keys and they are also empty if activation_similarity is empty
                                number_active_neurons[name] = self.batch_number_active_neurons[name]
                                active_classes_per_neuron[name] = self.batch_active_classes_per_neuron[name]
                                eval_dead_neurons[name] = self.batch_dead_neurons[name]
                            else:
                                number_active_neurons[name] = (number_active_neurons[name][0] + self.batch_number_active_neurons[name][0], 
                                                                    number_active_neurons[name][1]) 
                                # the total number of neurons is the same for all samples, hence we don't need to sum it up
                                active_classes_per_neuron[name] = active_classes_per_neuron[name] + self.batch_active_classes_per_neuron[name]
                                # dead_neurons is of the form [True,False,False,True,...] with size of the respective layer, where 
                                # "True" stands for dead neuron and "False" stands for "active" neuron. We have the previous dead_neurons 
                                # entry and a new one. A neuron is counted as dead if it was dead before and is still dead, 
                                # otherwise it is counted as "active". This can be achieved through pointwise multiplication.
                                eval_dead_neurons[name] = eval_dead_neurons[name] * self.batch_dead_neurons[name]
                        
                        for name in self.batch_activation_similarity.keys():
                            # the keys of self.batch_activation_similarity are the layer names instead of (layer_name,model_key) as above
                            if name not in activation_similarity:
                                activation_similarity[name] = self.batch_activation_similarity[name]
                            else:
                                activation_similarity[name] = activation_similarity[name] + self.batch_activation_similarity[name]
                ######## BATCH LOOP, USE SAE END ########

                # during training we log results after every batch/training step
                # recall that we don't print anything during training
                if train_or_eval == "train":
                    if self.use_sae:
                        print_and_log_results(train_or_eval=train_or_eval, 
                                            model_loss=batch_loss,
                                            accuracy=batch_accuracy,
                                            use_sae=self.use_sae,
                                            wandb_status=self.wandb_status,
                                            layer_names=self.layer_names,
                                            mean_number_active_classes_per_neuron=mean_number_active_classes_per_neuron,
                                            std_number_active_classes_per_neuron=std_number_active_classes_per_neuron,
                                            total_neurons=total_neurons,
                                            average_activated_neurons=average_activated_neurons,
                                            sparsity_dict_1=sparsity_dict_1,
                                            number_dead_neurons=number_dead_neurons,
                                            batch=self.batch_idx,
                                            sae_loss=self.batch_sae_loss, 
                                            sae_rec_loss=self.batch_sae_rec_loss,
                                            sae_l1_loss=self.batch_sae_l1_loss,
                                            sae_nrmse_loss=self.batch_sae_nrmse_loss,
                                            sae_rmse_loss=self.batch_sae_rmse_loss,
                                            kld=batch_kld,
                                            perc_same_classification=batch_perc_same_classification,
                                            activation_similarity=self.batch_activation_similarity)
                    else:
                        print_and_log_results(train_or_eval=train_or_eval, 
                                            model_loss=batch_loss,
                                            accuracy=batch_accuracy,
                                            use_sae=self.use_sae,
                                            wandb_status=self.wandb_status,
                                            layer_names=self.layer_names,
                                            batch=self.batch_idx)
                #tepoch.set_postfix(loss=)
        ######## BATCH LOOP END ########

        if train_or_eval == "eval":
            # once we have iterated over all batches, we average some quantities
            # by using drop_last=True when setting up the Dataloader (in utils.py) we make sure that 
            # the last batch of an epoch might has the same size as all other batches 
            # --> averaging by the number of batches is valid
            # the batch loss values are already means of the samples in the respective batch
            # hence it makes sense to also average the sum over the batches by the number of batches
            # to allow for better comparability
            num_batches = len(dataloader)
            model_loss = model_loss/num_batches
            accuracy = accuracy/num_batches
            if self.use_sae:
                sae_loss = sae_loss/num_batches
                sae_rec_loss = sae_rec_loss/num_batches
                sae_l1_loss = sae_l1_loss/num_batches
                sae_nrmse_loss = sae_nrmse_loss/num_batches
                sae_rmse_loss = sae_rmse_loss/num_batches
                kld = kld/num_batches
                perc_same_classification = perc_same_classification/num_batches
                activation_similarity = {k: (v[0]/num_batches, v[1]/num_batches) for k, v in activation_similarity.items()}
                number_active_neurons = {k: (v[0]/num_batches, v[1]) for k, v in number_active_neurons.items()}
                number_dead_neurons = compute_number_dead_neurons(eval_dead_neurons)
                sparsity_dict, sparsity_dict_1, number_active_classes_per_neuron, average_activated_neurons, total_neurons, mean_number_active_classes_per_neuron, std_number_active_classes_per_neuron = compute_sparsity(train_or_eval, 
                                                                                                self.sae_expansion_factor, 
                                                                                                number_active_neurons, 
                                                                                                active_classes_per_neuron, 
                                                                                                number_dead_neurons=number_dead_neurons,
                                                                                                dead_neurons=eval_dead_neurons, 
                                                                                                num_classes=self.num_classes)
                print_and_log_results(train_or_eval, 
                                        model_loss=model_loss,
                                        accuracy=accuracy,
                                        use_sae=self.use_sae,
                                        wandb_status=self.wandb_status,
                                        layer_names=self.layer_names,
                                        sparsity_dict=sparsity_dict,
                                        mean_number_active_classes_per_neuron=mean_number_active_classes_per_neuron,
                                        std_number_active_classes_per_neuron=std_number_active_classes_per_neuron,
                                        total_neurons=total_neurons,
                                        average_activated_neurons=average_activated_neurons,
                                        sparsity_dict_1=sparsity_dict_1,
                                        number_dead_neurons=number_dead_neurons,
                                        epoch=epoch,
                                        sae_loss=sae_loss,
                                        sae_rec_loss=sae_rec_loss,
                                        sae_l1_loss=sae_l1_loss,
                                        sae_nrmse_loss=sae_nrmse_loss,
                                        sae_rmse_loss=sae_rmse_loss,
                                        kld=kld,
                                        perc_same_classification=perc_same_classification,
                                        activation_similarity=activation_similarity)
            else:
                # if we don't use an SAE, we only log the model loss and accuracy
                print_and_log_results(train_or_eval, 
                                        model_loss=model_loss,
                                        accuracy=accuracy,
                                        use_sae=self.use_sae,
                                        wandb_status=self.wandb_status,
                                        layer_names=self.layer_names,
                                        epoch=epoch)
            
            # if we are in the last epoch, we store the sparsity dictionaries for access by another function
            if epoch == num_epochs and self.use_sae:
                self.sparsity_dict = sparsity_dict
                self.sparsity_dict_1 = sparsity_dict_1
                self.sae_rec_loss = sae_rec_loss
                self.sae_l1_loss = sae_l1_loss
                self.sae_nrmse_loss = sae_nrmse_loss
                self.sae_rmse_loss = sae_rmse_loss
                self.number_active_classes_per_neuron = number_active_classes_per_neuron
                self.active_classes_per_neuron = active_classes_per_neuron
            
        # We remove the hooks after every epoch. Otherwise, we will have 2 hooks for the next epoch.
        for hook in self.hooks:
            hook.remove()
        self.hooks = [] 

    def deploy_model(self, num_epochs):
        # if we are evaluating the modified model or the original model, we only perform one epoch
        if self.use_sae and not self.train_sae:
            num_epochs = 1
        elif not self.use_sae and not self.train_original_model:
            num_epochs = 1

        if self.use_sae:
            print("Using SAE...")
        else:
            print("Using the original model...")
        
        self.batch_idx = 0 # the batch_idx counts the total number of batches used during training across epochs
        self.train_dead_neurons = {}
 
        for epoch in range(num_epochs):
            if self.train_sae or self.train_original_model:
                # before the first training epoch we do one evaluation epoch on the test dataset
                if epoch==0: 
                    print("Doing one epoch of evaluation on the test dataset...")
                    self.epoch("eval", epoch, num_epochs)
                print("Doing one epoch of training...")
                self.epoch("train", epoch+1, num_epochs)

            # during evaluation and before every epoch of training we evaluate the model on the validation dataset
            print("Doing one epoch of evaluation on the test dataset...")
            self.epoch("eval", epoch+1, num_epochs)

        print("---------------------------")
        
        if self.use_sae:
            if self.train_sae:
                print("SAE training completed.")
                # store SAE model weights
                save_model_weights(self.sae_model, self.sae_weights_folder_path, layer_names=self.layer_names, params=self.sae_params)
            else:  
                print("Inference through the modified model completed.")
            # We store the sae_rec_loss and sae_l1_loss from the last epoch
            store_sae_eval_results(self.evaluation_results_folder_path, 
                                   self.layer_names, 
                                   self.sae_params_1, 
                                   self.sae_lambda_sparse, 
                                   self.sae_expansion_factor,
                                   self.sae_rec_loss, 
                                   self.sae_l1_loss,
                                   self.sae_nrmse_loss,
                                   self.sae_rmse_loss, 
                                   self.sparsity_dict, 
                                   self.sparsity_dict_1)
        elif self.train_original_model:
            print("Training of the original model completed.")
            # store original model weights
            save_model_weights(self.model, self.model_weights_folder_path, params=self.model_params)
        else:
            print("Inference through the original model completed.")

        params = {**self.model_params, **self.sae_params} # merge model_params and sae_params

        # We display the distribution of the number of classes that each neuron is active on, for the last epoch
        if self.use_sae:
            plot_active_classes_per_neuron(self.number_active_classes_per_neuron, 
                                        self.layer_names,
                                        num_classes=self.num_classes,
                                        folder_path=self.evaluation_results_folder_path, 
                                        params=params, 
                                        wandb_status=self.wandb_status)
            # We display the distribution of how active the neurons are, for the last epoch
            plot_neuron_activation_density(self.active_classes_per_neuron,
                                        self.layer_names,
                                        self.num_train_samples,
                                        folder_path=self.evaluation_results_folder_path, 
                                        params=params, 
                                        wandb_status=self.wandb_status)
            
        # We display some sample input images with their corresponding true and predicted labels as a sanity check 
        if self.wandb_status:
            log_image_table(self.val_dataloader,
                            self.category_names,
                            model=self.model, 
                            device=self.device)
        else:
            show_classification_with_images(self.val_dataloader,
                                            self.category_names,
                                            folder_path=self.evaluation_results_folder_path,
                                            model=self.model, 
                                            device=self.device,
                                            params=params)