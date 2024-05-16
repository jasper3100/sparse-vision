import torch 
from tqdm import tqdm
from torchvision import transforms
from utils import *
from get_sae_input_size import GetSaeInpSize
import copy
from einops import rearrange

class ModelPipeline:
    '''
    This class is used to perform the following tasks:
    - training the original model
    - perfoming inference through the original model
    - training the SAE
    - performing inference through the modified model (original model + SAE)
    - storing activations
    - returning statistics, such as losses, sparsity, accuracy, ...

    prof: torch.profiler.profile
        If not None, the profiler is used to profile the forward pass of the model to identify inefficiencies in the code.
    '''
    # constructor of the class (__init__ method)
    def __init__(self, 
                 device,
                 train_dataloader,
                 val_dataloader,
                 category_names,
                 sae_layer, 
                 wandb_status,
                 prof=None,
                 use_sae=None,
                 training=None, 
                 sae_weights_folder_path=None,
                 model_weights_folder_path=None,
                 evaluation_results_folder_path=None,
                 dead_neurons_steps=None,
                 sae_batch_size=None,
                 batch_size=None,
                 dataset_name=None,
                 directory_path=None,
                 mis=None): 
        self.device = device
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.category_names = category_names
        self.wandb_status = wandb_status
        #self.prof = prof 
        self.dead_neurons_steps = dead_neurons_steps
        self.sae_batch_size = sae_batch_size
        self.use_sae = use_sae
        self.training = training
        self.sae_layer = sae_layer
        self.dataset_name = dataset_name
        self.directory_path = directory_path
        self.mis = mis
    
        if self.use_sae:
            self.used_batch_size = sae_batch_size
        else:
            self.used_batch_size = batch_size

        # Folder paths
        self.sae_weights_folder_path = sae_weights_folder_path
        self.model_weights_folder_path = model_weights_folder_path
        self.evaluation_results_folder_path = evaluation_results_folder_path

        # Compute basic dataset statistics
        #self.num_train_samples = len(train_dataloader.dataset) # alternatively: train_dataset.__len__()
        #self.num_train_batches = len(train_dataloader)
        self.num_classes = len(category_names) # alternatively: len(train_dataloader.dataset.classes), but this is not always available
        # for example, for tiny imagenet we have to use category_names, as I had to obtain those manually in utils.py
        #self.num_eval_batches = len(val_dataloader)

        self.hooks = [] # list to store the hooks, 
        # so that we can remove them again later for using the model without hooks
        self.hooks1 = []

        self.record_top_samples = False # this will be set to True in the last epoch

        # MIS parameters (fixed here)
        k_mis = 9 # number of explanations (here: reference images, i.e., samples of the dataset (instead of f.e. feature visualizations))
        self.n_mis = 20 # number of tasks
        # self.k is used for getting the max & min self.k samples
        self.k = self.n_mis * (k_mis + 1) #200 #49#64 # number of top and small samples to record               

        if self.k > self.used_batch_size:
            self.n_mis = self.used_batch_size // (2*(k_mis + 1))
            # we do 2*(k_mis+1) because we want to get the top and small k samples and they shouldn't overlap
            # Example: 
            # self.used_batch_size = 15, then n_mis = 15 // 2*(9+1) = 0 --> not enough samples
            # self.used_batch_size = 20, then n_mis = 20 // 2*(9+1) = 1 --> 9 positive and 9 negative samples, and one query image for both cases
            # self.used_batch_size = 39, then n_mis = 40 // 2*(9+1) = 1 --> not enough samples for having 2 tasks (which require 40 images)
            if self.n_mis == 0:
                raise ValueError(f"Not enough samples in the batch to compute the MIS. Set batch_size at least to {2*(k_mis+1)}.")
            if self.n_mis != 20: 
                print(f"WARNING: Using {self.n_mis} tasks for computing the MIS (default is 20).")
            self.k = self.n_mis * (k_mis + 1)    

        self.get_histogram = False # this will be set to True when we want to get the histogram of the activations
        self.histogram_info = {}

        # we specify for which models (original, modified, sae) and layers we want to store the top/small k 
        # samples and generate the most activating image
        self.model_layer_list = []
        if self.use_sae:
            #for name in self.sae_layer: #.split("&"):
            #    if name != "":
            self.model_layer_list.extend([(self.sae_layer,'original'),(self.sae_layer,'modified'), (self.sae_layer,'sae')])
        #else:
        #    self.model_layer_list = [(self.sae_layer,'original')]

        # we specify for how many neurons in each layer we want to do the above
        self.number_neurons = 10 # make sure that it is not more than the maximal number of neurons in the desired layers

        # we get a dictionary mapping the image filenames to a corresponding index  
        # we do this once here, instead of for every batch in the process_batch fct. because this would be wasteful
        if self.dataset_name == "imagenet": 
            filename_txt = os.path.join(self.directory_path, 'dataloaders/imagenet_train_filenames.txt')
            self.filename_to_idx, self.idx_to_filename = get_string_to_idx_dict(filename_txt)
        else:
            self.filename_to_idx = None
            self.idx_to_filename = None

        
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
                           sae_params_1=None,
                           execution_location=None,
                           sae_checkpoint_epoch=None):
        self.model_name = model_name
        self.img_size = img_size
        self.sae_params = sae_params
        self.sae_params_1 = sae_params_1
        self.model_params = model_params
        self.sae_expansion_factor = sae_expansion_factor
        self.sae_lambda_sparse = sae_lambda_sparse
        self.sae_criterion_name = sae_criterion_name
        self.sae_optimizer_name = sae_optimizer_name
        self.sae_learning_rate = sae_learning_rate
        self.sae_checkpoint_epoch = sae_checkpoint_epoch
        self.model_criterion = get_criterion(model_criterion_name)
        if self.use_sae:
            self.sae_criterion = get_criterion(sae_criterion_name)

        # turn all values into string and merge them into a single string
        model_params_temp = {k: str(v) for k, v in self.model_params.items()}
        sae_params_temp = {k: str(v) for k, v in self.sae_params.items()}
        sae_params_1_temp = {k: str(v) for k, v in self.sae_params_1.items()} # used for post-hoc evaluation of several models wrt expansion factor, lambda sparse, learning rate,...
        self.params_string = '_'.join(model_params_temp.values()) + "_" + "_".join(sae_params_temp.values())
        self.params_string_1 = '_'.join(model_params_temp.values()) + "_" + "_".join(sae_params_1_temp.values()) # used for post-hoc evaluation

        # for checkpointing, we consider all params apart from the number of epochs, because we will checkpoint at specific custom epochs
        sae_params_temp.pop('sae_epochs', None)
        self.params_string_sae_checkpoint = '_'.join(model_params_temp.values()) + "_" + "_".join(sae_params_temp.values())
        
        ############### LOAD BASE MODEL ##################
        self.model = load_model(model_name, img_size=img_size, num_classes=self.num_classes)
        self.model = self.model.to(self.device)            
    
        if not self.use_sae and self.training: # train original model
            self.model_optimizer, self.optimizer_scheduler = get_optimizer(model_optimizer_name, self.model, model_learning_rate)   
        else: # inference through original model
            self.model.eval() # changes the behavior of certain layers, such as dropout
            # freeze model by disabling gradients
            for param in self.model.parameters():
                param.requires_grad = False
            # We don't specify above that the model is in training mode, because we do this 
            # in the epoch_forward_pass method where we first set it to train mode and then to eval mode to 
            # eval on the test set (this is done for every epoch)

        ############## LOAD SAE #######################
        if self.use_sae:
            self.sae_inp_size = GetSaeInpSize(self.model, self.sae_layer, self.train_dataloader, self.device, self.model_name).get_sae_inp_size()
            self.sae_model = load_model(sae_model_name, img_size=self.sae_inp_size, expansion_factor=sae_expansion_factor)
            if sae_checkpoint_epoch > 0:
                file_path = get_file_path(self.sae_weights_folder_path, self.sae_layer, params=self.params_string_sae_checkpoint, file_name= f'sae_checkpoint_epoch_{sae_checkpoint_epoch}.pth')
                checkpoint = torch.load(file_path)
                state_dict = checkpoint['model_state_dict']
                self.train_batch_idx = checkpoint['training_step'] # the batch_idx counts the total number of batches used during training across epochs
                '''
                # WHY ARE WE DOING THE BELOW???
                if "W_enc" in state_dict:
                    print("W_enc in state dict")
                    # take the transpose of the weight matrix
                    state_dict["encoder.weight"] = state_dict.pop("W_enc").T
                if "b_enc" in state_dict:
                    print("b_enc in state dict")
                    state_dict["encoder.bias"] = state_dict.pop("b_enc")
                if "W_dec" in state_dict:
                    print("W_dec in state dict")
                    state_dict["decoder.weight"] = state_dict.pop("W_dec").T
                if "b_dec" in state_dict:
                    print("b_dec in state dict")
                    state_dict["decoder.bias"] = state_dict.pop("b_dec")
                '''
                self.sae_model.load_state_dict(state_dict)
                print(f"Use SAE on layer {self.sae_layer} from epoch {sae_checkpoint_epoch}")
            elif sae_checkpoint_epoch == 0:
                self.train_batch_idx = 0
            self.sae_model = self.sae_model.to(self.device)
            
            if self.training:
                self.sae_optimizer, _ = get_optimizer(sae_optimizer_name, self.sae_model, sae_learning_rate)
                if sae_checkpoint_epoch > 0:
                    self.sae_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    print("Optimizer state loaded")
            else:
                self.sae_model.eval()
                for param in self.sae_model.parameters():
                    param.requires_grad = False

            # if we are using an SAE, we also create a copy of the original model so that we have 2 models
            # one modified model (with SAE) + one original model --> enables us to compare the outputs of those models
            # This model is always used in inference mode only
            self.model_copy = copy.deepcopy(self.model) # using load_pretrained_model might not give exactly the same model!
            self.model_copy = self.model_copy.to(self.device)
            self.model_copy.eval()
            for param in self.model_copy.parameters():
                param.requires_grad = False

    def compute_and_store_batch_wise_metrics(self, model_key, output, name, expansion_factor=1, output_2=None):
        '''
        This computes and logs certain metrics for each batch.

        Parameters
        ----------
        model_key : str
            "original" if not use_sae (layer outputs of original model)
            "modified" if use_sae (layer outputs of modified model)
            "sae" (encoder outputs of SAE)
        output : torch.Tensor, shape: [batch_size, num_neurons in respective layer]
        name : str
            The name of the current layer of the model
        '''
        # for certain computations we use the spatial mean of conv activations
        output_avg_W_H, output_2_avg_W_H = average_over_W_H(output, output_2)
            
        if self.get_histogram:
            # for the histogram we use the average activation per channel, i.e., modified output and modified output 2
            self.histogram_info = update_histogram(self.histogram_info, name, model_key, output_avg_W_H, self.device, output_2=output_2_avg_W_H)
        else:
            batch_dead_units, batch_sparsity = measure_inactive_units(output, expansion_factor)
            # since a unit is inactive if it's 0, it doesn't matter whether we look at the post- (output) or pre-relu encoder output (output_2)
            self.batch_sparsity[(name,model_key)] = batch_sparsity # float item and thus on cpu
            #self.batch_number_active_neurons[(name,model_key)] = (number_active_neurons, number_total_neurons)
            batch_dead_units = batch_dead_units.detach()
            self.batch_dead_units[(name,model_key)] = batch_dead_units.cpu()
            #print("Shape batch dead units:", batch_dead_units.shape)

            # store a matrix of size [#neurons, #classes] with one entry for each neuron (of the current layer) and each class, with the number of how often this neuron 
            # is active on samples from that class for the current batch; measure of polysemanticity
            # THE BELOW COMPUTATION WOULD HAVE TO BE ADJUSTED FOR CONV LAYER OUTPUTS (in particular, targets would have to be reshaped somehow, 
            # but since I don't really use this measure, I don't adjust it for now
            ''' # is this correct? It still gives an error...
            if len(output.shape) == 4:
                targets = np.repeat(self.targets, output.size(2)*output.size(3))
                # absolute output has shape [BS*H*W, C], targets should have shape [BS*H*W]
                assert targets.shape[0] == absolute_output.shape[0]      
            else: 
                targets = self.targets
            active_classes_per_neuron = active_classes_per_neuron_aux(absolute_output, targets, self.num_classes, self.activation_threshold)
            self.batch_active_classes_per_neuron[(name,model_key)] = active_classes_per_neuron
            '''

            if self.record_top_samples:                
                # we get the top k samples for each neuron in the current batch and the corresponding
                # indices (in the batch) of those samples --> the two matrices below both have shape
                # [k, #neurons] and [k, #neurons]

                # One special case occurs if k > batch size. In this case, in the first run/batch, the matrix
                # will have a dimension of batch size < k. The matrix will be extended in size until it has size k, i.e., 
                # the desired size [k, #neurons]. This is done later when the matrices of two batches are merged. 

                # if we use an sae we only compute the top samples of the SAE layer and not of the original or modified layer
                if (self.use_sae and  model_key == "sae") or not self.use_sae:

                    # we use the channel average activations for getting the top k samples

                    if output_2 is not None: # if we consider the SAE, then we look at the prerelu encoder output
                        # otherwise, the smallest values will be 0 --> but activation histogram is also for prerelu
                        # encoder output --> need smallest values prerelu!!!
                        use_output = output_2_avg_W_H
                    else:
                        use_output = output_avg_W_H

                    self.batch_top_k_values[(name,model_key)] = torch.topk(use_output, k=self.k, dim=0)[0]
                    self.batch_top_k_indices[(name,model_key)] = torch.topk(use_output, k=self.k, dim=0)[1]
                    self.batch_small_k_values[(name,model_key)] = torch.topk(use_output, k=self.k, dim=0, largest=False)[0]
                    self.batch_small_k_indices[(name,model_key)] = torch.topk(use_output, k=self.k, dim=0, largest=False)[1]


    def hook(self, module, input, output, name, use_sae, train_sae):
        '''
        Retrieve and possibly modify outputs of the original model
        Shape of variable output: [channels, height, width] --> no batch dimension, since we iterate over each batch
        '''  
        output = output.detach() # doesn't seem to have an effect, intuition: we don't need gradients of the original model output if we attach a hook
        # if we want to train the original model, we don't use a hook anyways

        # we store quantities of the original model
        if not use_sae:
            self.compute_and_store_batch_wise_metrics(model_key='original', output=output, name=name)
        
        # use the sae to modify the output of the specified layer of the original model
        if use_sae and name == self.sae_layer:
            sae_model = self.sae_model

            # the output of the specified layer of the original model is the input of the SAE
            # if the output has 4 dimensions, we flatten it to 2 dimensions along the scheme: (BS, C, H, W) -> (BS*W*H, C)
            if len(output.shape) == 4:
                modified_output = rearrange(output, 'b c h w -> (b h w) c')
                transformed = True
            else: # if the output has 2 dimensions, we just keep it as it is          
                modified_output = output  
                transformed = False    
                      
            sae_input = modified_output

            if train_sae and name == self.sae_layer:
                with torch.enable_grad():  # Enable gradients
                    encoder_output, decoder_output, encoder_output_prerelu = sae_model(sae_input) 
                    if transformed:
                        encoder_output = rearrange(encoder_output, '(b h w) c -> b c h w', b=output.size(0), h=output.size(2), w=output.size(3))
                        encoder_output_prerelu = rearrange(encoder_output_prerelu, '(b h w) c -> b c h w', b=output.size(0), h=output.size(2), w=output.size(3))
                        # note that the encoder_output(_prerelu) has c*k channels, where k is the expansion factor    
                    rec_loss, l1_loss, nrmse_loss, rmse_loss = self.sae_criterion(encoder_output, decoder_output, sae_input) # the sae inputs are the targets
                    loss = rec_loss + self.sae_lambda_sparse * l1_loss
                    self.sae_optimizer.zero_grad(set_to_none=True) # sae optimizer only has gradients of SAE
                    sae_model.zero_grad(set_to_none=True)
                    loss.backward()
                    self.sae_optimizer.step()
                    self.sae_optimizer.zero_grad(set_to_none=True)
                    sae_model.zero_grad(set_to_none=True)        
            else:
                with torch.no_grad():
                    encoder_output, decoder_output, encoder_output_prerelu = sae_model(sae_input)
                    if transformed:
                        encoder_output = rearrange(encoder_output, '(b h w) c -> b c h w', b=output.size(0), h=output.size(2), w=output.size(3))
                        encoder_output_prerelu = rearrange(encoder_output_prerelu, '(b h w) c -> b c h w', b=output.size(0), h=output.size(2), w=output.size(3))
                        # note that the encoder_output(_prerelu) has c*k channels, where k is the expansion factor            
                    rec_loss, l1_loss, nrmse_loss, rmse_loss = self.sae_criterion(encoder_output, decoder_output, sae_input)
                    loss = rec_loss + self.sae_lambda_sparse*l1_loss

            self.batch_sae_rec_loss[name] = rec_loss.item()
            self.batch_sae_l1_loss[name] = l1_loss.item()
            self.batch_sae_loss[name] = loss.item()
            self.batch_sae_nrmse_loss[name] = nrmse_loss.item()
            self.batch_sae_rmse_loss[name] = rmse_loss.item()
            
            encoder_output = encoder_output.detach() # doesn't seem to have an effect
            encoder_output_prerelu = encoder_output_prerelu.detach() # doesn't seem to have an effect
            decoder_output = decoder_output.detach() # has an effect! because we pass the decoder output 
            # back to the model, we have to remove the gradients here, otherwise we accumulate 
            # unnecessary gradients in the rest of the model --> eventually might lead to out of memory error

            if encoder_output.grad is not None:
                print("Encoder output has gradients.")
            if encoder_output_prerelu.grad is not None:
                print("Encoder output prerelu has gradients.")
            if decoder_output.grad is not None:
                print("Decoder output has gradients.")

            # store quantities of the encoder output
            self.compute_and_store_batch_wise_metrics(model_key='sae', output=encoder_output, name=name, expansion_factor=self.sae_expansion_factor, output_2 = encoder_output_prerelu)

            if len(output.shape) == 4:                
                decoder_output = rearrange(decoder_output, '(b h w) c -> b c h w', b=output.size(0), h=output.size(2), w=output.size(3))
                assert decoder_output.shape == output.shape
            self.batch_var_expl[name] = variance_explained(output, decoder_output).item()

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

    def register_hooks(self, use_sae, train_sae, model, model_copy=None):
        # we register hooks on layers of the original model
        # split self.sae_layer based on &
        # for now, we only register hooks on the layer on which we train or use an SAE
        for name in self.sae_layer.split("&"):
            if name != "":
                m = None
                #print(name)
                # separate name based on "."
                for subname in name.split("."):
                    if m is None:
                        m = getattr(model, subname)
                    else: 
                        m = getattr(m, subname)
                #print(m)
                # for example, for 'layer1.0.conv1' -->  m = getattr(model, 'layer1') --> m = getattr(m, '0') --> m = getattr(m, 'conv1')
                hook = m.register_forward_hook(lambda module, inp, out, name=name, use_sae=use_sae, train_sae=train_sae: self.hook(module, inp, out, name, use_sae, train_sae))
                # manually: hook = model.layer1[0].conv1.register_forward_hook(lambda module, inp, out, name='layer1.0.conv1', use_sae=use_sae, train_sae=train_sae: self.hook(module, inp, out, name, use_sae, train_sae))
                self.hooks.append(hook)

                # we do the same for the model copy if we use the SAE
                if use_sae:
                    m = None 
                    for subname in name.split("."):
                        if m is None:
                            m = getattr(model_copy, subname)
                        else: 
                            m = getattr(m, subname)
                    hook1 = m.register_forward_hook(lambda module, inp, out, name=name: self.hook_2(module, inp, out, name))
                    self.hooks1.append(hook1)
            '''
            # Alternatively, register hooks on all modules
            #module_names = get_module_names(model)
            for name, m in model.named_children():
                #m = getattr(model, name)
                hook = m.register_forward_hook(lambda module, inp, out, name=name, use_sae=use_sae, train_sae=train_sae: self.hook(module, inp, out, name, use_sae, train_sae))
                self.hooks.append(hook)
                if use_sae: # see method description of hook_2 for an explanation on what this is doing
                    m1 = getattr(model_copy, name)
                    hook1 = m1.register_forward_hook(lambda module, inp, out, name=name: self.hook_2(module, inp, out, name))
                    self.hooks1.append(hook1)
            '''

    def epoch(self, epoch_mode, epoch, num_epochs):
        '''
        epoch_mode | self.use_sae | 
        "train"       | False        | train the original model
        "train"       | True         | train the SAE
        "eval"        | False        | evaluate the original model
        "eval"        | True         | evaluate the modified model        
        '''
        if epoch_mode == "train":
            dataloader = self.train_dataloader
            if not self.use_sae: # original model
                # set model to train mode and unfreeze parameters
                self.model.train()
                for param in self.model.parameters():
                    param.requires_grad = True
                train_sae = False
            else: # modified model / train SAE
                self.sae_model.train()
                for param in self.sae_model.parameters():
                    param.requires_grad = True
                train_sae = True

        elif epoch_mode == "eval" or epoch_mode == "mis":
            train_sae = False

            if epoch_mode == "eval":
                dataloader = self.val_dataloader
            elif epoch_mode == "mis": # for computing the MIS we use the train dataset
                # we get the max and min activating images so it doesn't matter if train dataset is shuffled
                dataloader = self.train_dataloader 

            # set model to eval mode and freeze parameters
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False
            if hasattr(self, 'sae_model'):
                self.sae_model.eval()
                for param in self.sae_model.parameters():
                    param.requires_grad = False

            if epoch == num_epochs or epoch_mode == "mis":
                # if we are in the last epoch (and in eval mode), we record the top
                # samples of each neuron, i.e., the samples activating each neuron the most
                # (and the least)
                self.record_top_samples = True

        if self.use_sae:
            self.register_hooks(self.use_sae, train_sae, self.model, self.model_copy) # registering the hook within the batches for loop will lead to undesired behavior
            # as the hook will be registered multiple times --> activations will be captured multiple times!
        else:
            #pass
            self.register_hooks(self.use_sae, train_sae, self.model)
            
        '''
        # During training, we perform data augmentation
        if epoch_mode == "train":
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
        '''

        #if epoch_mode == "train":
            #augmentations_list = []

        # if we are in eval mode, we keep track of quantities across batches
        if epoch_mode == "eval":
            accuracy = 0.0
            model_loss = 0.0
            loss_diff = 0.0
            #if epoch <= num_epochs: # in the additional epoch for creating the histograms
            #number_active_neurons = {}
            #active_classes_per_neuron = {}
            eval_dead_neurons = {}
            sparsity_dict = {}
            sae_loss = {}
            sae_rec_loss = {}
            sae_l1_loss = {}
            sae_nrmse_loss = {}
            sae_rmse_loss = {}
            var_expl = {}
            perc_same_classification = 0.0
            kld = 0.0
            #activation_similarity = {}
            perc_eval_dead_neurons = {}
            #self.activations = {}
            correct_count_by_class = {} # we only need this once to find out the classes on which the model classifies the best
            total_count_by_class = {} # we only need this once to find out the classes on which the model classifies the best
        
        if epoch_mode == "eval" or epoch_mode == "mis":
            self.top_k_samples = {}
            self.small_k_samples = {}
            eval_batch_idx = 0

        #print("Before batch: Memory Allocated (in bytes):", torch.cuda.memory_allocated(), "Memory Reserved (in bytes):", torch.cuda.memory_reserved())

        ######## BATCH LOOP START ########
        with tqdm(dataloader, unit="batch") as tepoch:
            for batch in tepoch:
                tepoch.set_description(f'{epoch_mode} epoch {epoch}')
        #for batch in dataloader:
                #self.batch_activations = {}
                #self.batch_number_active_neurons = {}
                self.batch_sparsity = {}
                self.batch_dead_units = {}
                #self.batch_active_classes_per_neuron = {}
                #self.batch_activation_similarity = {}
                self.batch_sae_rec_loss = {}
                self.batch_sae_l1_loss = {}
                self.batch_sae_loss = {}
                self.batch_sae_nrmse_loss = {}
                self.batch_sae_rmse_loss = {}
                self.batch_var_expl = {}
                perc_train_dead_neurons = {}

                if epoch_mode == "eval" or epoch_mode == "mis":
                    self.batch_top_k_values = {}
                    self.batch_top_k_indices = {}
                    self.batch_small_k_values = {}
                    self.batch_small_k_indices = {}
                
                inputs, targets, filename_indices = process_batch(batch, 
                                                                epoch_mode=epoch_mode, 
                                                                filename_to_idx=self.filename_to_idx) 
                # filename_indices is torch tensor of indices (1-dim tensor)
                inputs, self.targets, filename_indices = inputs.to(self.device), targets.to(self.device), filename_indices.to(self.device)

                if self.dataset_name == "imagenet":
                    #self.targets_original = self.targets.clone()
                    label_translator = get_label_translator(self.directory_path)
                    # maps Pytorch imagenet labels to labelling convention used by InceptionV1
                    self.targets = label_translator(self.targets) # f.e. tensor([349, 898, 201,...]), i.e., tensor of label indices
                    # this translation is necessary because the InceptionV1 model return output indices in a different convention
                    # hence, for computing the accuracy f.e. we need to translate the Pytorch indices to the InceptionV1 indices

                if epoch_mode == "train" and not self.use_sae: # train original model
                    outputs = self.model(inputs)
                    batch_loss = self.model_criterion(outputs, self.targets)
                    self.model_optimizer.zero_grad()
                    batch_loss.backward()
                    self.model_optimizer.step()
                    self.model_optimizer.zero_grad()
                    self.model.zero_grad() # hence we should also set gradients of model to zero?
                else:
                    with torch.no_grad(): # TRY THIS
                        outputs = self.model(inputs)
                        batch_loss = self.model_criterion(outputs, self.targets)

                batch_loss = batch_loss.item() # .item() automatically moves this to CPU
                _, class_ids = torch.max(outputs, 1)
                batch_accuracy = torch.sum(class_ids == self.targets).item() / self.targets.size(0)
                # batch_accuracy is a float item and thus on CPU by default

                ''' # we only need this once to find out the classes on which the model classifies the best
                for gt_class_idx, i in zip(self.targets, range(len(self.targets))): # gt = ground truth

                    if gt_class_idx.item() in total_count_by_class:
                        total_count_by_class[gt_class_idx.item()] += 1
                    else:
                        total_count_by_class[gt_class_idx.item()] = 1

                    if gt_class_idx.item() == class_ids[i].item():
                        # class_ids is the predicted class index
                        if gt_class_idx.item() in correct_count_by_class:
                            correct_count_by_class[gt_class_idx.item()] += 1
                        else:
                            correct_count_by_class[gt_class_idx.item()] = 1
                '''

                if self.use_sae:
                    # We use (a copy of) the original model to get its outputs and activations
                    # and then compare them with those of the modified model
                    # hook_2 should be registered on self.model_copy to get the activations
                    with torch.no_grad():
                        outputs_original = self.model_copy(inputs)
                        batch_loss_original = self.model_criterion(outputs_original, self.targets).item()
                        batch_loss_diff = batch_loss - batch_loss_original
                        # we apply first softmax (--> prob. distr.) and then log
                        log_prob_original = F.log_softmax(outputs_original, dim=1).cpu()
                        log_prob_modified = F.log_softmax(outputs, dim=1).cpu()
                        # log_target = True means that we pass log(target) instead of target (second argument of kl_div)
                        batch_kld = F.kl_div(log_prob_original, log_prob_modified, reduction='sum', log_target=True).item()
                        # see the usage example in https://pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html
                        batch_kld = batch_kld / inputs.size(0) # we divide by the batch size
                        # we calculate the percentage of same classifications of modified and original model
                        _, class_ids_original = torch.max(outputs_original, dim=1)
                        #class_ids_original = class_ids_original.to(self.device)
                        batch_perc_same_classification = (class_ids_original == class_ids).sum().item() / class_ids_original.size(0)

                    # we calculate the feature similarity between the modified and original model
                    # self.batch_activation_similarity = {}###feature_similarity(self.batch_activations,self.batch_activation_similarity,self.device)
                else: # if not use self sae
                    batch_kld = 0.0
                    batch_perc_same_classification = 0.0
                    #self.batch_activation_similarity = {}

                ######## COMPUTE QUANTITIES ########
                if epoch_mode == "train":
                    ''' # there are 2 options to perform data augmentations: here with randomly picking a transformation
                    # or when loading the data at the beginning
                    # transform each img to torch.uint8
                    #for img in inputs:
                    #    img.type(torch.uint8)
                    inputs = inputs.type(torch.uint8)
                    # Randomly choose one augmentation from the list for each image in the batch
                    random_augmentation = transforms.RandomChoice(augmentations_list)
                    # Apply the randomly chosen augmentation to each image in the batch
                    inputs = torch.stack([random_augmentation(img) for img in inputs])
                    # apply toTensor to each img
                    #inputs = torch.stack([transforms.ToTensor()(img) for img in inputs])
                    # apply normalization to each img
                    inputs = inputs.type(torch.float32)
                    inputs = torch.stack([transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))(img) for img in inputs])
                    '''
                    # the batch idx only counts batches used during training
                    self.train_batch_idx += 1
                    #if self.train_batch_idx == 3:
                    #    break
                    # if we are in train mode we keep track of dead neurons possibly across several epochs
                    # depending on the value of dead_neurons_steps
                    for name in self.batch_dead_units.keys():
                        if name not in self.train_dead_neurons:
                            self.train_dead_neurons[name] = self.batch_dead_units[name] # shape: [#units]
                        else:
                            self.train_dead_neurons[name] = self.train_dead_neurons[name] * self.batch_dead_units[name]
                        # in train mode, we measure dead neurons for every batch
                        perc_train_dead_neurons[name] = self.train_dead_neurons[name].sum().item() / (self.train_dead_neurons[name].shape[0])
                        # we don't get sparsity_dict here because it depends on the number of dead neurons, which changes during training
                        # --> we don't measure sparsity (definition 0) during training
                        # _, sparsity_dict_1, number_active_classes_per_neuron, mean_number_active_classes_per_neuron, std_number_active_classes_per_neuron = compute_sparsity(epoch_mode, 
                        #_, sparsity_dict_1, _, _, _ = compute_sparsity(epoch_mode, self.sae_expansion_factor,self.batch_number_active_neurons,num_classes=self.num_classes)
                                                                                                #active_classes_per_neuron=self.batch_active_classes_per_neuron,
                    '''
                    During training, we also re-initialize dead neurons, which were dead over the last n steps (n=dead_neurons_steps)
                    Then, we let the model train with the new neurons for n steps
                    Then, we measure dead neurons for another n steps and re-initialize the dead neurons from those last n steps etc.
                    When using the original model, we just measure dead neurons and set the counter back to zero every n steps but we don't re-initialize dead neurons
                    '''
                    if self.train_batch_idx % self.dead_neurons_steps == 0 and (self.train_batch_idx // self.dead_neurons_steps) % 2 == 1:
                        # first condition: self.train_batch_idx is a multiple of self.dead_neurons_steps
                        # second condition: self.train_batch_idx is an odd multiple of self.dead_neurons_steps
                        # --> self.train_batch_idx = n* self.dead_neurons_steps where n = 1,3,5,7,...

                        # we re-initialize dead neurons in the SAE (recall that the base model is assumed to be given 
                        # and frozen) by initializing the weights into a dead neuron (i.e. in the encoder)
                        # with a Kaiming uniform distribution and we set the corresponding bias to zero 
                        # this might be one of the most basic ways possible to re-initialize dead neurons

                        # get the dead neurons of the SAE encoder output
                        # (so far we assume that sae_layer is a list with only one element)
                        if self.use_sae:
                            pass
                            #######dead_neurons_sae = self.train_dead_neurons[(self.sae_layer, 'sae')]
                            #self.sae_model.reset_encoder_weights(dead_neurons_sae=dead_neurons_sae, device=self.device, optimizer=self.sae_optimizer, batch_idx=self.train_batch_idx)

                        # we start measuring dead neurons from 0
                        self.train_dead_neurons = {}

                    elif self.train_batch_idx % self.dead_neurons_steps == 0 and (self.train_batch_idx // self.dead_neurons_steps) % 2 == 0 and self.train_batch_idx != 0:
                        # --> self.train_batch_idx = n* self.dead_neurons_steps where n = 2,4,6,8,... (the last condition excludes n=0)
                        self.train_dead_neurons = {}

                # if we are in eval mode we we accumulate batch-wise quantities
                if epoch_mode == "eval":
                    model_loss += batch_loss
                    accuracy += batch_accuracy

                    for name in self.batch_sparsity.keys():
                        if name not in eval_dead_neurons:
                            # the below dictionaries have the same keys and they are also empty if activation_similarity is empty
                            #number_active_neurons[name] = self.batch_number_active_neurons[name]
                            sparsity_dict[name] = self.batch_sparsity[name]
                            #active_classes_per_neuron[name] = self.batch_active_classes_per_neuron[name]
                            eval_dead_neurons[name] = self.batch_dead_units[name]
                        else:
                            sparsity_dict[name] = sparsity_dict[name] + self.batch_sparsity[name]
                            #number_active_neurons[name] = (number_active_neurons[name][0] + self.batch_number_active_neurons[name][0], 
                            #                                    number_active_neurons[name][1]) 
                            # the total number of neurons is the same for all samples, hence we don't need to sum it up
                            ###active_classes_per_neuron[name] = active_classes_per_neuron[name] + self.batch_active_classes_per_neuron[name]
                            # dead_neurons is of the form [True,False,False,True,...] with size of the respective layer, where 
                            # "True" stands for dead neuron and "False" stands for "active" neuron. We have the previous dead_neurons 
                            # entry and a new one. A neuron is counted as dead if it was dead before and is still dead, 
                            # otherwise it is counted as "active". This can be achieved through pointwise multiplication.
                            eval_dead_neurons[name] = eval_dead_neurons[name] * self.batch_dead_units[name]
                    #print("Eval dead neurons:", eval_dead_neurons)
                                    
                    if epoch == num_epochs:
                        self.targets = self.targets.cpu()
                        if not hasattr(self, 'targets_epoch'):
                            self.targets_epoch = self.targets
                        else:
                            self.targets_epoch = torch.cat((self.targets_epoch, self.targets), dim=0)

                    if self.use_sae:
                        for name in self.batch_sae_rec_loss.keys():
                            if name not in sae_loss:
                                sae_rec_loss[name] = self.batch_sae_rec_loss[name]
                                sae_l1_loss[name] = self.batch_sae_l1_loss[name]
                                sae_loss[name] = self.batch_sae_loss[name]
                                sae_nrmse_loss[name] = self.batch_sae_nrmse_loss[name]
                                sae_rmse_loss[name] = self.batch_sae_rmse_loss[name]
                                var_expl[name] = self.batch_var_expl[name]
                            else:
                                sae_rec_loss[name] += self.batch_sae_rec_loss[name]
                                sae_l1_loss[name] += self.batch_sae_l1_loss[name]
                                sae_loss[name] += self.batch_sae_loss[name]
                                sae_nrmse_loss[name] += self.batch_sae_nrmse_loss[name]
                                sae_rmse_loss[name] += self.batch_sae_rmse_loss[name]
                                var_expl[name] += self.batch_var_expl[name]
                        kld += batch_kld
                        perc_same_classification += batch_perc_same_classification
                        loss_diff += batch_loss_diff

                        '''
                        for name in self.batch_activation_similarity.keys():
                            # the keys of self.batch_activation_similarity are the layer names instead of (layer_name,model_key) as above
                            if name not in activation_similarity:
                                activation_similarity[name] = self.batch_activation_similarity[name]
                            else:
                                activation_similarity[name] = activation_similarity[name] + self.batch_activation_similarity[name]
                        '''

                if epoch_mode == "eval" or epoch_mode == "mis":
                    for name in self.batch_top_k_values.keys():
                        if name not in self.top_k_samples.keys(): 
                            self.top_k_samples[name] = (self.batch_top_k_values[name], 
                                                        self.batch_top_k_indices[name], # shape [k, #neurons]
                                                        self.used_batch_size, 
                                                        filename_indices[self.batch_top_k_indices[name]])
                            # filename_indices is a 1-dim tensor of batch size 
                            # -> filename_indices[self.batch_top_k_indices[name]] is a tensor of shape [k, #neurons]
                            self.small_k_samples[name] = (self.batch_small_k_values[name], 
                                                            self.batch_small_k_indices[name], 
                                                            self.used_batch_size,
                                                            filename_indices[self.batch_small_k_indices[name]])
                            #self.activations[name] = self.batch_activations[name]
                        else:
                            self.top_k_samples[name] = get_top_k_samples(self.top_k_samples[name], 
                                                                    self.batch_top_k_values[name], 
                                                                    self.batch_top_k_indices[name],
                                                                    filename_indices[self.batch_top_k_indices[name]],
                                                                    eval_batch_idx,
                                                                    largest=True,
                                                                    k=self.k)
                            self.small_k_samples[name] = get_top_k_samples(self.small_k_samples[name], 
                                                                    self.batch_small_k_values[name], 
                                                                    self.batch_small_k_indices[name],
                                                                    filename_indices[self.batch_small_k_indices[name]],
                                                                    eval_batch_idx,
                                                                    largest=False,
                                                                    k=self.k)
                            #self.activations[name] = torch.cat((self.activations[name], self.batch_activations[name]), dim=0) # dim=0 is the batch dimension

                    eval_batch_idx += 1
                    #if eval_batch_idx == 5:
                    #    torch.cuda.memory._dump_snapshot(os.path.join(self.directory_path, f"memory_dump_{epoch_mode}_{epoch}.pth"))
                    #    break

                # during training we log results after every batch/training step
                # recall that we don't print anything during training
                if epoch_mode == "train":
                    print_and_log_results(epoch_mode=epoch_mode, 
                                        model_loss=batch_loss,
                                        loss_diff=batch_loss_diff,
                                        accuracy=batch_accuracy,
                                        use_sae=self.use_sae,
                                        wandb_status=self.wandb_status,
                                        sparsity_dict=self.batch_sparsity,
                                        #mean_number_active_classes_per_neuron=mean_number_active_classes_per_neuron,
                                        #std_number_active_classes_per_neuron=std_number_active_classes_per_neuron,
                                        #number_active_neurons=self.batch_number_active_neurons,
                                        #sparsity_dict_1=sparsity_dict_1,
                                        perc_dead_neurons=perc_train_dead_neurons,
                                        batch=self.train_batch_idx,
                                        sae_loss=self.batch_sae_loss, 
                                        sae_rec_loss=self.batch_sae_rec_loss,
                                        sae_l1_loss=self.batch_sae_l1_loss,
                                        sae_nrmse_loss=self.batch_sae_nrmse_loss,
                                        sae_rmse_loss=self.batch_sae_rmse_loss,
                                        var_expl=self.batch_var_expl,
                                        kld=batch_kld,
                                        perc_same_classification=batch_perc_same_classification)
                                        #activation_similarity=self.batch_activation_similarity)
                #tepoch.set_postfix(loss=)
                # if we are running the code locally and if the dataset is not MNIST, 
                # we break after the first batch, because we are only testing the code
                if self.device == torch.device('cpu') and self.dataset_name != 'mnist':
                    break
                        
        ######## BATCH LOOP END ########

        # move all tensors to the CPU to save GPU memory, [2] --> batch size --> int --> already on cpu                         
        #self.top_k_samples[name] = (self.top_k_samples[name][0].cpu(), self.top_k_samples[name][1].cpu(), self.top_k_samples[name][2], self.top_k_samples[name][3].cpu())
        #self.small_k_samples[name] = (self.small_k_samples[name][0].cpu(), self.small_k_samples[name][1].cpu(), self.small_k_samples[name][2], self.small_k_samples[name][3].cpu())
        #print("Device batch top k values", self.batch_top_k_values[name].device)

        # we store max and min filename indices for computing the MIS later (separately)
        if self.top_k_samples != {}:
            folder_path = os.path.join(self.evaluation_results_folder_path, 'filename_indices')
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            for name in self.top_k_samples.keys():
                data = {'max_filename_indices': self.top_k_samples[name][3], 'min_filename_indices': self.small_k_samples[name][3]} # shape of each tensor: [k, #neurons]
                layer_name = name[0]
                model_key = name[1]
                file_path = get_file_path(folder_path=folder_path,
                                        sae_layer=model_key + '_' + layer_name,
                                        params=self.params_string,
                                        file_name=f'max_min_filename_indices_epoch_{epoch}.pt')
                torch.save(data, file_path)

        if not self.use_sae and epoch_mode == 'train': # train original model
            if self.optimizer_scheduler is not None:
                self.optimizer_scheduler.step()

        if epoch_mode == "eval":
            # once we have iterated over all batches, we average some quantities
            # by using drop_last=True when setting up the Dataloader (in utils.py) we make sure that 
            # the last batch of an epoch might has the same size as all other batches 
            # --> averaging by the number of batches is valid
            # the batch loss values are already means of the samples in the respective batch
            # hence it makes sense to also average the sum over the batches by the number of batches
            # to allow for better comparability
            num_batches = eval_batch_idx # len(dataloader) doesn't work with webdataset/imagenet
            model_loss = model_loss/num_batches
            loss_diff = loss_diff/num_batches
            accuracy = accuracy/num_batches
            for name in sae_loss.keys():
                sae_loss[name] = sae_loss[name]/num_batches
                sae_rec_loss[name] = sae_rec_loss[name]/num_batches
                sae_l1_loss[name] = sae_l1_loss[name]/num_batches
                sae_nrmse_loss[name] = sae_nrmse_loss[name]/num_batches
                sae_rmse_loss[name] = sae_rmse_loss[name]/num_batches
                var_expl[name] = var_expl[name]/num_batches
            kld = kld/num_batches
            perc_same_classification = perc_same_classification/num_batches
            for name in sparsity_dict.keys():
                sparsity_dict[name] = sparsity_dict[name]/num_batches
                # eval_dead_neurons[name] has shape [#units]
                perc_eval_dead_neurons[name] = eval_dead_neurons[name].sum().item() / eval_dead_neurons[name].size(0) # sum of dead units / # units
                #print(eval_dead_neurons[name].shape[0])
                #print(eval_dead_neurons[name].shape[1])
                #print(eval_dead_neurons[name].sum().item())
                #print(perc_eval_dead_neurons[name])
            #activation_similarity = {k: (v[0]/num_batches, v[1]/num_batches) for k, v in activation_similarity.items()}
            #number_active_neurons = {k: (v[0]/num_batches, v[1]) for k, v in number_active_neurons.items()}
            #sparsity_dict, sparsity_dict_1, number_active_classes_per_neuron, mean_number_active_classes_per_neuron, std_number_active_classes_per_neuron = compute_sparsity(epoch_mode, 
            #sparsity_dict, sparsity_dict_1, _, _, _ = compute_sparsity(epoch_mode,self.sae_expansion_factor,number_active_neurons,number_dead_neurons=number_dead_neurons,
            #                                                           dead_neurons=eval_dead_neurons,num_classes=self.num_classes)
                                                                        #active_classes_per_neuron=active_classes_per_neuron, 

            ''' # we only need this once to find out the classes on which the model classifies the best
            accuracy_by_class = {}
            for class_index in total_count_by_class.keys():
                if self.dataset_name == "imagenet":
                    adj_class_idx = class_index - 1
                    # we do class_index - 1 because the class indices start at 1 but the list of names starts with index 0
                    # for the dictonaries we don't go by the key indices but by the actual keys which correspond to class_index
                else:
                    adj_class_idx = class_index
                accuracy_by_class[self.category_names[adj_class_idx]] = correct_count_by_class[class_index] / total_count_by_class[class_index]
            # Sort the dictionary by values in descending order
            sorted_dict = sorted(accuracy_by_class.items(), key=lambda x: x[1], reverse=True)
            file_name = os.path.join(self.directory_path, 'accuracy_by_class_imagenet_trainset.txt')
            with open(file_name, 'w') as f:
                for key, value in sorted_dict:
                    f.write(f"{key} {value}\n")
            '''
            
            if self.training:
                # after each eval epoch during training, we store some SAE results
                # We store the sae_rec_loss and sae_l1_loss from the last epoch
                store_sae_eval_results(self.evaluation_results_folder_path, 
                                    self.sae_layer, 
                                    self.params_string_1, 
                                    epoch,
                                    self.sae_lambda_sparse, 
                                    self.sae_expansion_factor,
                                    self.sae_batch_size,
                                    self.sae_optimizer_name,
                                    self.sae_learning_rate,
                                    sae_rec_loss[self.sae_layer], 
                                    sae_l1_loss[self.sae_layer],
                                    sae_nrmse_loss[self.sae_layer],
                                    sae_rmse_loss[self.sae_layer], 
                                    sparsity_dict[(self.sae_layer,'sae')],
                                    var_expl[self.sae_layer],
                                    perc_eval_dead_neurons[(self.sae_layer,'sae')],
                                    loss_diff,
                                    median_mis=0)
                                    #self.sparsity_dict_1[(name,'sae')])


            if epoch <= num_epochs: # if we do the extra eval epoch for getting the activation histogram we don't want to print or log anything                              
                print_and_log_results(epoch_mode, 
                                        model_loss=model_loss,
                                        loss_diff=loss_diff,
                                        accuracy=accuracy,
                                        use_sae=self.use_sae,
                                        wandb_status=self.wandb_status,
                                        sparsity_dict=sparsity_dict,
                                        #mean_number_active_classes_per_neuron=mean_number_active_classes_per_neuron,
                                        #std_number_active_classes_per_neuron=std_number_active_classes_per_neuron,
                                        #number_active_neurons=number_active_neurons,
                                        #sparsity_dict_1=sparsity_dict_1,
                                        perc_dead_neurons=perc_eval_dead_neurons,
                                        epoch=epoch,
                                        sae_loss=sae_loss,
                                        sae_rec_loss=sae_rec_loss,
                                        sae_l1_loss=sae_l1_loss,
                                        sae_nrmse_loss=sae_nrmse_loss,
                                        sae_rmse_loss=sae_rmse_loss,
                                        var_expl=var_expl,
                                        kld=kld,
                                        perc_same_classification=perc_same_classification,
                                        #activation_similarity=activation_similarity)
                                        median_mis=0)
                
            #'''
            if epoch > num_epochs:
                # after the extra evaluation epoch for computing the activation histogram, we create the input image which activates a certain neuron the most 
                # we do this at the very end because we modify self.model and the forward function of the SAE
                if self.use_sae:
                    # for obtaining the maximally activating example, we create a new model by combining the layers of the original model
                    # and the SAE model, so that we can access the SAE layers directly

                    # TO-DO ADJUST TO THE CASE OF MULTIPLE SAE'S!!!!
                    if self.training:
                        sae_model = self.sae_model
                    else:
                         # for now we assume that we only use one SAE
                        assert len(self.pretrained_saes_list) == 1
                        for pretrain_sae_name in self.pretrained_saes_list:
                            # the last layer of the pretrain_sae_name is the layer on which the SAE was trained
                            # for example, "fc1&fc2&fc3" --> pretrain_sae_layer_name = "fc3"
                            sae_model = getattr(self, f"model_sae_{pretrain_sae_name}")
                            sae_model = sae_model.to(self.device)
                            sae_model = sae_model.eval()
                            for param in sae_model.parameters():
                                param.requires_grad = False

                    # We get the output of self.model with hook to later double-check that 
                    # the new modified model gives the same output 
                    for batch in self.val_dataloader:
                        inputs, targets, _ = process_batch(batch)                     
                        inputs, self.targets = inputs.to(self.device), targets.to(self.device)

                        if self.dataset_name == "imagenet":
                            #self.targets_original = self.targets.clone()
                            label_translator = get_label_translator(self.directory_path)
                            # maps Pytorch imagenet labels to labelling convention used by InceptionV1
                            self.targets = label_translator(self.targets) # f.e. tensor([349, 898, 201,...]), i.e., tensor of label indices
                            
                        outputs_1 = self.model(inputs)
                        break
                
                    # we define a new forward function for the SAE, which is as the original forward fct. except
                    # that it only returns the decoder output --> so that inference through the model is possible
                    def new_forward(self, x):
                        if len(x.shape) == 4:
                            x_new = rearrange(x, 'b c h w -> (b h w) c')   
                            transformed = True
                        else:
                            transformed = False
                            x_new = x
                        x_cent = x_new - self.decoder.bias
                        encoder_output_prerelu = self.encoder(x_cent)
                        encoder_output = self.sae_act(encoder_output_prerelu)
                        decoder_output = self.decoder(encoder_output)
                        if transformed:
                            decoder_output = rearrange(decoder_output, '(b h w) c -> b c h w', b=x.size(0), h=x.size(2), w=x.size(3))
                            assert decoder_output.shape == x.shape
                        return decoder_output

                    # add new_forward function to the SAE model instance as a class method
                    # we have to do this after running self.model, because self.model still uses the original forward function
                    setattr(sae_model, 'forward', new_forward.__get__(sae_model, sae_model.__class__))

                    # remove hooks from self.model
                    for hook in self.hooks:
                        hook.remove()
                    self.hooks = [] 

                    # we add the SAE
                    # for some reason, it doesn't properly work when using self.model_copy 
                    # with Resnet-18 this gives an error from lucent about "There are no saved feature maps" --> maybe 
                    # using copy to get self.model_copy is somehow incompatible...(?)
                    for name in self.sae_layer.split("&"):
                        if name != "":
                            m = None
                            # separate name based on "."
                            if len(name.split(".")) > 1:
                                for subname in name.split(".")[:-1]:
                                    if m is None:
                                        m = getattr(self.model, subname)
                                    else: 
                                        m = getattr(m, subname)
                                setattr(m, name.split(".")[-1], nn.Sequential(getattr(m, name.split(".")[-1]), sae_model))
                                # Explanation: F.e. for 'layer1.0.conv1' -->  m = getattr(model, 'layer1') --> m = getattr(m, '0') 
                                # --> and then name.split(".")[-1] = 'conv1' 
                                # --> so we do setattr(self.model.layer1.0, "conv1", nn.Sequential(self.model.layer1.0.conv1, sae_model))
                                # which corresponds to self.model.layer1[0].conv1 = nn.Sequential(self.model.layer1[0].conv1, sae_model)
                                # Apparently setattr only works by taking an object and a non-empty string, also we can't do "layer1[0].conv1" 
                                # since the string can only contain the name of one individual object and not a list of objects
                            elif len(name.split(".")) == 1:
                                setattr(self.model, name, nn.Sequential(getattr(self.model, name), sae_model))
                                # Otherwise m would be None above
                            else:
                                raise ValueError("Unexpected layer name format")
                
                    outputs = self.model(inputs)   
                    # check if the outputs are the same
                    assert torch.allclose(outputs, outputs_1, atol=1e-5), "The outputs of the original model and the modified model are not the same."

                #print(get_model_layers(self.model))
                # If we use an SAE, the new layer names are f.e.: 'layer1_0_conv1' (the overall layer), with sublayers:
                # 'layer1_0_conv1_0' (the original conv layer) 
                # SAE: 'layer1_0_conv1_1' (x_cent in SAE MLP), 'layer1_0_conv1_1_encoder', 'layer1_0_conv1_1_sae_act', 'layer1_0_conv1_1_decoder'

                '''
                plot_lucent_explanations(self.model, 
                                            self.sae_layer, 
                                            self.params_string, 
                                            self.evaluation_results_folder_path, 
                                            self.wandb_status,
                                            self.number_neurons)
                '''
            #'''

            ''' # this only works for MNIST reasonably well
            for layer_name, model_key in self.model_layer_list:
                self.create_maximally_activating_images(layer_name, 
                                                        model_key, 
                                                        self.top_k_samples, 
                                                        self.evaluation_results_folder_path, 
                                                        self.sae_layer, 
                                                        self.params_string, 
                                                        self.number_neurons, 
                                                        self.wandb_status) 
            '''   

        # We remove the hooks after every epoch. Otherwise, we will have 2 hooks for the next epoch.
        for hook in self.hooks:
            hook.remove()
        self.hooks = [] 
        for hook in self.hooks1:
            hook.remove()
        self.hooks1 = []

        if epoch_mode == "train" and self.use_sae:
            # we create a checkpoint after every training epoch
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.sae_model.state_dict(),
                'optimizer_state_dict': self.sae_optimizer.state_dict(),
                'training_step': self.train_batch_idx,
            }
            os.makedirs(self.sae_weights_folder_path, exist_ok=True) # create folder if it doesn't exist
            file_path = get_file_path(self.sae_weights_folder_path, self.sae_layer, self.params_string_sae_checkpoint, f'sae_checkpoint_epoch_{epoch}.pth')
        
            start_time = time.time()
            torch.save(checkpoint, file_path)
            print("Time to save SAE model weights:", time.time() - start_time)
            print(f"Successfully stored checkpoint in {file_path}")

        # we empty the GPU memory to reduce the chances of getting a CUDA out of memory error
        if self.device == torch.device('cuda'):
            gc.collect()
            torch.cuda.empty_cache()

    def deploy_model(self, num_epochs):
        if self.use_sae:
            print(f"Using SAE...")
        else:
            print("Using the original model...")

        self.train_dead_neurons = {}    
        #with torch.autograd.profiler.profile(use_cuda=True) as prof:
        #'''
        if self.use_sae:
            # we start at sae_checkpoint_epoch and train up to num_epochs
            start = self.sae_checkpoint_epoch
        else: # train original model
            start = 0 

        if self.training:
            for epoch in range(start, num_epochs):
                #prof.step()
            
                if self.training:
                    # before the first training epoch we do one evaluation epoch on the test dataset
                    #'''
                    ###if epoch==0: 

                    print("Doing one epoch of EVALUATION on the test dataset...\n")
                    #with torch.autograd.profiler.profile(use_cuda=True) as prof1:
                    self.epoch("eval", epoch, num_epochs)
                    #print(prof1.key_averages().table(sort_by="cuda_time_total"))
                    #'''
                    #with torch.autograd.profiler.profile(use_cuda=True) as prof1:
                    print("Doing one epoch of TRAINING...\n")
                    #torch.cuda.memory._record_memory_history(max_entries=100000)
                    self.epoch("train", epoch+1, num_epochs)
                    #torch.cuda.memory._dump_snapshot(os.path.join(self.directory_path, f"memory_dump_add_memory_on_purpose.pth"))
            
                    #print(prof1.key_averages().table(sort_by="cuda_time_total"))
                
                # during evaluation and before every epoch of training we evaluate the model on the validation dataset
                print("Doing one epoch of EVALUATION on the test dataset...\n")
                #print("Before eval: Memory Allocated (in bytes):", torch.cuda.memory_allocated(), "Memory Reserved (in bytes):", torch.cuda.memory_reserved())
                #with torch.autograd.profiler.profile(use_cuda=True, profile_memory=True) as prof1:
                self.epoch("eval", epoch+1, num_epochs)
                ######self.prof.export_memory_timeline(f"stuff.html", device="cuda:0")
                #print(prof1.key_averages().table(sort_by="self_cuda_memory_usage"))#sort_by="cuda_time_total"))
            #print(prof.key_averages().table(sort_by="cuda_time_total"))
            
            #torch.cuda.memory._record_memory_history(enabled=None)
        elif not self.training and self.mis == "0":
            self.epoch("eval", start, start+1) # the exact value of num_epochs (here start+1) doesn't matter, but we just don't want the case where epoch > num_epochs, hence we just do +1 here
        elif not self.training and self.mis == "1":
            # we compute the max and min k samples and store the filename indices to be used for computing the MIS
            self.epoch("mis", start, start+1)
        elif not self.training and self.mis == "2":
            # we compute the mis
            if self.use_sae:
                model_key = "sae"
            else:
                model_key = "original"
            compute_mis(evaluation_results_folder_path=self.evaluation_results_folder_path, 
                        params_string=self.params_string, 
                        model_key=model_key, 
                        layer_name=self.sae_layer, 
                        idx_to_filename=self.idx_to_filename, 
                        n_mis=self.n_mis, 
                        epoch=start)

        print("---------------------------")

        #'''
        if self.use_sae:
            aux_string = "SAE"
        else:
            aux_string = "Original model"

        if self.training: 
            print(f"Training of the {aux_string} completed.")
        if self.mis == "0":
            print(f"Inference through the {aux_string} model completed.")
        if self.mis == "1":
            print(f"{aux_string}: Storing values for computing the MIS completed.")
        if self.mis == "2":
            print(f"{aux_string}: Computing the MIS completed.")
        #'''

        # We display some sample input images with their corresponding true and predicted labels as a sanity check 
        '''
        show_classification_with_images(self.val_dataloader,
                                        self.category_names,
                                        self.wandb_status,
                                        self.directory_path,
                                        folder_path=self.evaluation_results_folder_path,
                                        model=self.model, 
                                        device=self.device,
                                        params=self.params_string)
        '''

        ''' # DO NOT USE THESE ON TINY IMAGENET (in general they are a bit outdated)
        # We display the distribution of the number of classes that each neuron is active on, for the last epoch
        if self.use_sae:
            plot_active_classes_per_neuron(self.number_active_classes_per_neuron, 
                                        self.sae_layer,
                                        num_classes=self.num_classes,
                                        folder_path=self.evaluation_results_folder_path, 
                                        params=self.params_string, 
                                        wandb_status=self.wandb_status)
            # We display the distribution of how active the neurons are, for the last epoch
            plot_neuron_activation_density(self.active_classes_per_neuron,
                                        self.sae_layer,
                                        self.num_train_samples,
                                        folder_path=self.evaluation_results_folder_path, 
                                        params=self.params_string, 
                                        wandb_status=self.wandb_status)
        '''

        '''
        # show the most highly activating samples for a specific neuron of a specific layer of a specific model
        for layer_name, model_key in self.model_layer_list:
            show_top_k_samples(self.val_dataloader, 
                            model_key=model_key, 
                            layer_name=layer_name, 
                            top_k_samples=self.top_k_samples, 
                            small_k_samples=self.small_k_samples,
                            folder_path=self.evaluation_results_folder_path,
                            sae_layer=self.sae_layer,
                            params=self.params_string,
                            n=int(np.sqrt(self.k)),
                            number_neurons=self.number_neurons,
                            wandb_status=self.wandb_status,
                            dataset_name=self.dataset_name)
        '''

        # generate activation histogram
        '''
        print("---------------------------")
        print("Doing an extra round of inference to get the activation histogram...")
        num_bins = 100

        for key, v in self.top_k_samples.items():
            # top_k_samples is a tuple where the top values are the first element of the tuple
            top_k_values = v[0]
            # top_k_values has shape [k, #units] --> we get the first row to get the top values 
            top_values = top_k_values[0,:]
            # we only look at the first n units
            self.number_neurons = min(self.number_neurons, top_values.shape[0])
            top_values = top_values[:self.number_neurons]

            # we do the same for the smallest activation values
            small_k_values = self.small_k_samples[key][0]
            small_values = small_k_values[0,:]
            small_values = small_values[:self.number_neurons]

            # we create an empty matrix to store the histogram values later on
            histogram_matrix = torch.zeros(num_bins, self.number_neurons)
            self.histogram_info[key] = (histogram_matrix, top_values, small_values)

        # then, we loop through the model and put the activations into the histogram matrix based on the bins
        # which we infer from the top and small values
        self.get_histogram = True
        self.epoch("eval", num_epochs + 1, num_epochs) # we do an additional round of evaluation
        activation_histograms_2(self.histogram_info, 
                              folder_path=self.evaluation_results_folder_path,
                              sae_layer=self.sae_layer, 
                              params=self.params_string, 
                              wandb_status=self.wandb_status, 
                              num_units=self.number_neurons)
        '''

        ''' # outdated as we don't store activations anymore
        activation_histograms(self.activations,
                              folder_path=self.evaluation_results_folder_path,
                              sae_layer=self.sae_layer,
                              params=self.params_string,
                              wandb_status=self.wandb_status)
        # we plot the same histogram but distinguishing by classes
        activation_histograms(self.activations,
                              folder_path=self.evaluation_results_folder_path,
                              sae_layer=self.sae_layer,
                              params=self.params_string,
                              wandb_status=self.wandb_status,
                              targets=self.targets_epoch)
        '''