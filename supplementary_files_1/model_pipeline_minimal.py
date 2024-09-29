import torch 
from tqdm import tqdm
from utils import *
from get_sae_input_size import GetSaeInpSize
from einops import rearrange

# this modelpipeline is only for training an SAE on Imagenet!!!

class ModelPipeline:
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
                 training=None, 
                 sae_weights_folder_path=None,
                 model_weights_folder_path=None,
                 evaluation_results_folder_path=None,
                 dead_neurons_steps=None,
                 sae_batch_size=None,
                 batch_size=None,
                 dataset_name=None,
                 directory_path=None): 
        self.device = device
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.category_names = category_names
        self.activation_threshold = activation_threshold
        self.wandb_status = wandb_status
        self.dead_neurons_steps = dead_neurons_steps
        self.sae_batch_size = sae_batch_size
        self.use_sae = use_sae
        self.training = training
        self.layer_names = layer_names
        self.dataset_name = dataset_name
        self.directory_path = directory_path
        self.model_weights_folder_path = model_weights_folder_path
        self.used_batch_size = sae_batch_size
        self.num_classes = len(category_names) 
        self.hooks = [] 

        # we get a dictionary mapping the image filenames to a corresponding index  
        # we do this once here, instead of for every batch in the process_batch fct. because this would be wasteful
        filename_txt = os.path.join(self.directory_path, 'dataloaders/imagenet_train_filenames.txt')
        self.filename_to_idx, self.idx_to_filename = get_string_to_idx_dict(filename_txt)

        
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
                           execution_location=None):
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
        self.model_criterion = get_criterion(model_criterion_name)

        self.model = load_pretrained_model(model_name,
                                            img_size,
                                            self.model_weights_folder_path,
                                            num_classes=self.num_classes,
                                            params=self.model_params,
                                            execution_location=execution_location)
        self.model = self.model.to(self.device)
        # If we don't train the original model we only use it to perform inference,
        # hence we do the following: We set it to eval mode (changes the behavior of 
        # certain layers, such as dropout)
        self.model.eval()
        # and we freeze the model by disabling gradients
        for param in self.model.parameters():
            param.requires_grad = False

        #self.sae_criterion = #get_criterion(sae_criterion_name, sae_lambda_sparse)
        self.train_sae_layer = self.layer_names
        setattr(self, f"sae_{self.train_sae_layer}_inp_size", GetSaeInpSize(self.model, self.train_sae_layer, self.train_dataloader, self.device, self.model_name).get_sae_inp_size())
        temp_sae_inp_size = getattr(self, f"sae_{self.train_sae_layer}_inp_size")
        self.sae_model = load_model(sae_model_name, img_size=temp_sae_inp_size, expansion_factor=sae_expansion_factor).to(self.device)
        
        self.sae_optimizer, _ = get_optimizer(sae_optimizer_name, self.sae_model, sae_learning_rate)

    def hook(self, module, input, output, name, use_sae, train_sae):
        '''
        Retrieve and possibly modify outputs of the original model
        Shape of variable output: [channels, height, width] --> no batch dimension, since we iterate over each batch
        '''  
        output = output.detach() # doesn't seem to have an effect, intuition: we don't need gradients of the original model output if we attach a hook
        # if we want to train the original model, we don't use a hook anyways

        # use the sae to modify the output of the specified layer of the original model
        if use_sae and name == self.train_sae_layer:
            # the output of the specified layer of the original model is the input of the SAE
            # if the output has 4 dimensions, we flatten it to 2 dimensions along the scheme: (BS, C, H, W) -> (BS*W*H, C)
            if len(output.shape) == 4:
                modified_output = rearrange(output, 'b c h w -> (b h w) c')
                transformed = True
            else: # if the output has 2 dimensions, we just keep it as it is          
                modified_output = output  
                transformed = False    
                      
            sae_input = modified_output

            if train_sae:
                with torch.enable_grad():  # Enable gradients
                    encoder_output, decoder_output, encoder_output_prerelu = self.sae_model(sae_input) 
                    if transformed:
                        encoder_output = rearrange(encoder_output, '(b h w) c -> b c h w', b=output.size(0), h=output.size(2), w=output.size(3))
                        encoder_output_prerelu = rearrange(encoder_output_prerelu, '(b h w) c -> b c h w', b=output.size(0), h=output.size(2), w=output.size(3))
                        # note that the encoder_output(_prerelu) has c*k channels, where k is the expansion factor    
                    #rec_loss, l1_loss, nrmse_loss, rmse_loss = self.sae_criterion(encoder_output, decoder_output, sae_input) # the sae inputs are the targets
                    l1_loss = 0
                    rec_loss = nn.MSELoss()(decoder_output, sae_input)
                    loss = rec_loss + self.sae_lambda_sparse*l1_loss
                    self.sae_optimizer.zero_grad(set_to_none=True) # sae optimizer only has gradients of SAE
                    self.sae_model.zero_grad(set_to_none=True)
                    loss.backward()
                    self.sae_optimizer.step()
                    self.sae_optimizer.zero_grad(set_to_none=True)
                    self.sae_model.zero_grad(set_to_none=True)        
            else:
                with torch.no_grad():
                    encoder_output, decoder_output, encoder_output_prerelu = self.sae_model(sae_input)
                    if transformed:
                        encoder_output = rearrange(encoder_output, '(b h w) c -> b c h w', b=output.size(0), h=output.size(2), w=output.size(3))
                        encoder_output_prerelu = rearrange(encoder_output_prerelu, '(b h w) c -> b c h w', b=output.size(0), h=output.size(2), w=output.size(3))
                        # note that the encoder_output(_prerelu) has c*k channels, where k is the expansion factor            
                    #rec_loss, l1_loss, _, _ = self.sae_criterion(encoder_output, decoder_output, sae_input)
                    l1_loss = 0
                    rec_loss = nn.MSELoss()(decoder_output, sae_input)
                    loss = rec_loss + self.sae_lambda_sparse*l1_loss

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

            if len(output.shape) == 4:                
                decoder_output = rearrange(decoder_output, '(b h w) c -> b c h w', b=output.size(0), h=output.size(2), w=output.size(3))
                assert decoder_output.shape == output.shape

            # we pass the decoder_output back to the original model
            output = decoder_output

            loss = None 
            encoder_output = None
            decoder_output = None
            encoder_output_prerelu = None
            sae_input = None

            # below line doesn't seem to have an effect
            del encoder_output, decoder_output, encoder_output_prerelu, sae_input, loss, rec_loss, l1_loss

            #gc.collect()
            #torch.cuda.empty_cache()
        return output
    
    def register_hooks(self, use_sae, train_sae, model, model_copy=None):
        m = getattr(model, self.train_sae_layer)
        hook = m.register_forward_hook(lambda module, inp, out, name=self.train_sae_layer, use_sae=use_sae, train_sae=train_sae: self.hook(module, inp, out, name, use_sae, train_sae))
        # manually: hook = model.layer1[0].conv1.register_forward_hook(lambda module, inp, out, name='layer1.0.conv1', use_sae=use_sae, train_sae=train_sae: self.hook(module, inp, out, name, use_sae, train_sae))
        self.hooks.append(hook)

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
            self.sae_model.train()
            for param in self.sae_model.parameters():
                param.requires_grad = True
            train_sae = True

        elif epoch_mode == "eval":
            dataloader = self.val_dataloader
            self.sae_model.eval()
            for param in self.sae_model.parameters():
                param.requires_grad = False
            train_sae = False
          
        self.register_hooks(self.use_sae, train_sae, self.model) # registering the hook within the batches for loop will lead to undesired behavior
        # as the hook will be registered multiple times --> activations will be captured multiple times!
    
        label_translator = get_label_translator(self.directory_path)

        ######## BATCH LOOP START ########
        with tqdm(dataloader, unit="batch") as tepoch:
            for batch in tepoch:
                tepoch.set_description(f'{epoch_mode} epoch {epoch}')
                
                inputs, targets, filename_indices = process_batch(batch, 
                                                                epoch_mode=epoch_mode, 
                                                                filename_to_idx=self.filename_to_idx) 
                # filename_indices is torch tensor of indices (1-dim tensor)
                inputs, self.targets, filename_indices = inputs.to(self.device), targets.to(self.device), filename_indices.to(self.device)

                if self.dataset_name == "imagenet":
                    #self.targets_original = self.targets.clone()
                    # maps Pytorch imagenet labels to labelling convention used by InceptionV1
                    self.targets = label_translator(self.targets) # f.e. tensor([349, 898, 201,...]), i.e., tensor of label indices
                    # this translation is necessary because the InceptionV1 model return output indices in a different convention
                    # hence, for computing the accuracy f.e. we need to translate the Pytorch indices to the InceptionV1 indices

                with torch.no_grad():
                    outputs = self.model(inputs)
                
                '''
                inputs = inputs.detach()
                outputs = outputs.detach()
                self.targets = self.targets.detach()
                filename_indices = filename_indices.detach()

                inputs = None
                self.targets = None
                filename_indices = None 

                del inputs, self.targets, filename_indices

                gc.collect()
                torch.cuda.empty_cache()
                '''
        ######## BATCH LOOP END ########

        for hook in self.hooks:
            hook.remove()
        self.hooks = [] 
     
        if self.device == torch.device('cuda'):
            gc.collect()
            torch.cuda.empty_cache()

    def deploy_model(self, num_epochs):
        for epoch in range(num_epochs):
            if epoch==0: 
                print("Doing one epoch of evaluation on the test dataset...")
                self.epoch("eval", epoch, num_epochs)
            print("Doing one epoch of training...")
            self.epoch("train", epoch+1, num_epochs)
            print("Doing one epoch of evaluation on the test dataset...")
            self.epoch("eval", epoch+1, num_epochs)