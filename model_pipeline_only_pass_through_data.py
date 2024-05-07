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
        self.prof = prof 
        self.dead_neurons_steps = dead_neurons_steps
        self.sae_batch_size = sae_batch_size
        self.use_sae = use_sae
        self.training = training
        self.layer_names = layer_names
        self.dataset_name = dataset_name
        self.directory_path = directory_path
    
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

        self.record_top_samples = True # False # this will be set to True in the last epoch

        # MIS parameters (fixed here)
        k_mis = 9 # number of explanations (here: reference images, i.e., samples of the dataset (instead of f.e. feature visualizations))
        self.n_mis = 20 # number of tasks

        # self.k is used for getting the max & min self.k samples
        self.k = self.n_mis * (k_mis + 1) #200 #49#64 # number of top and small samples to record

        self.get_histogram = False # this will be set to True when we want to get the histogram of the activations
        self.histogram_info = {}

        # we specify for which models (original, modified, sae) and layers we want to store the top/small k 
        # samples and generate the most activating image
        self.model_layer_list = []
        if self.use_sae:
            for name in self.layer_names.split("_"):
                if name != "":
                    self.model_layer_list.extend([(name,'original'),(name,'modified'), (name,'sae')])
       # else:
       #     self.model_layer_list = [('fc1','original')]

        # we specify for how many neurons in each layer we want to do the above
        self.number_neurons = 10 # make sure that it is not more than the maximal number of neurons in the desired layers

        if self.dataset_name == "imagenet": 
            # we get a dictionary mapping the image filenames to a corresponding index      
            filename_txt = os.path.join(self.directory_path, 'dataloaders/imagenet_train_val_filenames.txt')
            self.filename_to_idx_val, self.idx_to_filename_val = get_string_to_idx_dict(filename_txt)

            if self.training:
                filename_txt = os.path.join(self.directory_path, 'dataloaders/imagenet_train_train_filenames.txt')
                self.filename_to_idx_train, self.idx_to_filename_train = get_string_to_idx_dict(filename_txt)
            else:
                self.filename_to_idx_train = None
                self.idx_to_filename_train = None
        else:
            self.filename_to_idx_val = None
            self.filename_to_idx_train = None
            self.idx_to_filename_val = None
            self.idx_to_filename_train = None
        
        #print("filename_to_idx_val:", self.filename_to_idx_val)
        #print("filename_to_idx_train:", self.filename_to_idx_train)
        #print("idx_to_filename_val:", self.idx_to_filename_val)
        #print("idx_to_filename_train:", self.idx_to_filename_train)

        
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

        # turn all values into string and merge them into a single string
        model_params_temp = {k: str(v) for k, v in self.model_params.items()}
        sae_params_temp = {k: str(v) for k, v in self.sae_params.items()}
        sae_params_1_temp = {k: str(v) for k, v in self.sae_params_1.items()} # used for post-hoc evaluation of several models wrt expansion factor, lambda sparse, learning rate,...
        self.params_string = '_'.join(model_params_temp.values()) + "_" + "_".join(sae_params_temp.values())
        self.params_string_1 = '_'.join(model_params_temp.values()) + "_" + "_".join(sae_params_1_temp.values()) # used for post-hoc evaluation

        if not self.use_sae and self.training: # train original model --> use fresh model
            self.model = load_model(model_name, img_size=img_size, num_classes=self.num_classes, execution_location=execution_location)
            self.model = self.model.to(self.device)
            self.model_optimizer, self.optimizer_scheduler = get_optimizer(model_optimizer_name, self.model, model_learning_rate)
            # We don't specify whether the model is in training or evaluation mode here, because we do this 
            # in the epoch_forward_pass method where we first set it to train mode and then to eval mode to 
            # eval on the test set (this is done for every epoch)
        else: # inference through original model or if we use sae --> use pretrained model
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

        if self.use_sae:
            if self.training:
                # split self.layer_names on the last occurence of "_"
                if "_" in self.layer_names:
                    pretrained_sae_layers_string, self.train_sae_layer = self.layer_names.rsplit('_', 1)
                else:
                    pretrained_sae_layers_string = ""
                    self.train_sae_layer = self.layer_names
                if pretrained_sae_layers_string == "":
                    print("Training SAE on layer", self.train_sae_layer, "and not using any pretrained SAEs.")
                else:
                    print("Training SAE on layer", self.train_sae_layer, "and using pretrained SAEs on layers", pretrained_sae_layers_string)
            else:
                if self.layer_names.endswith("__"):
                    pretrained_sae_layers_string = self.layer_names[:-2]
            
            # Split the string into a list based on '_'
            self.pretrained_sae_layers_list = pretrained_sae_layers_string.split("_")

            # since the SAE models are trained sequentially, we need to load the pretrained SAEs in the correct order
            # f.e. if pretrained_sae_layers_list = ["layer1", "layer2", "layer3", "layer4"], then we need to load the 
            # SAEs with names: "layer1", "layer1_layer2", "layer1_layer2_layer3", "layer1_layer2_layer3_layer4"
            self.pretrained_saes_list = []
            for i in range(len(self.pretrained_sae_layers_list)):
                joined = '_'.join(self.pretrained_sae_layers_list[:i+1])
                self.pretrained_saes_list.append(joined)

            self.sae_criterion = get_criterion(sae_criterion_name, sae_lambda_sparse)

            if pretrained_sae_layers_string != "":

                for name in self.pretrained_sae_layers_list:
                    setattr(self, f"sae_{name}_inp_size", GetSaeInpSize(self.model, name, self.train_dataloader, self.device, self.model_name).get_sae_inp_size())
                # if we use pretrained SAEs, we load all of them
                for pretrain_sae_name in self.pretrained_saes_list:
                    # the last layer of the pretrain_sae_name is the layer on which the SAE was trained
                    # for example, "fc1_fc2_fc3" --> pretrain_sae_layer_name = "fc3"
                    pretrain_sae_layer_name = pretrain_sae_name.split("_")[-1]
                    temp_sae_inp_size = getattr(self, f"sae_{pretrain_sae_layer_name}_inp_size")
                    setattr(self, f"model_sae_{pretrain_sae_name}", load_pretrained_model(sae_model_name,
                                                                                        temp_sae_inp_size,
                                                                                        self.sae_weights_folder_path,
                                                                                        sae_expansion_factor=sae_expansion_factor,
                                                                                        layer_name=pretrain_sae_name,
                                                                                        params=self.params_string))
                    sae_model = getattr(self, f"model_sae_{pretrain_sae_name}")
                    sae_model = sae_model.to(self.device)
                    sae_model = sae_model.eval()
                    for param in sae_model.parameters():
                        param.requires_grad = False
                    print("Loaded pretrained SAE model on layer", pretrain_sae_name)
                                
            # we instantiate a fresh SAE for the layer on which we want to train the SAE
            if self.training:    
                setattr(self, f"sae_{self.train_sae_layer}_inp_size", GetSaeInpSize(self.model, self.train_sae_layer, self.train_dataloader, self.device, self.model_name).get_sae_inp_size())
                temp_sae_inp_size = getattr(self, f"sae_{self.train_sae_layer}_inp_size")
                self.sae_model = load_model(sae_model_name, img_size=temp_sae_inp_size, expansion_factor=sae_expansion_factor).to(self.device)
                self.sae_optimizer, _ = get_optimizer(sae_optimizer_name, self.sae_model, sae_learning_rate)

            # if we are using an SAE, we also create a copy of the original model so that we have 2 models
            # one modified model (with SAE) + one original model --> enables us to compare the outputs of those models
            # This model is always used in inference mode only
            self.model_copy = copy.deepcopy(self.model) # using load_pretrained_model might not give exactly the same model!
            self.model_copy = self.model_copy.to(self.device)
            self.model_copy.eval()
            for param in self.model_copy.parameters():
                param.requires_grad = False

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
            else: # modified model
                # set model to eval mode and unfreeze parameters
                self.sae_model.train()
                for param in self.sae_model.parameters():
                    param.requires_grad = True
                train_sae = True

        elif epoch_mode == "eval":
            dataloader = self.val_dataloader
            if not self.use_sae: # original model
                # set model to eval mode and freeze parameters
                self.model.eval()
                for param in self.model.parameters():
                    param.requires_grad = False
                train_sae = False
            else:
                # set model to eval mode and freeze parameters
                # if self.sae_model exists
                if hasattr(self, 'sae_model'):
                    self.sae_model.eval()
                    for param in self.sae_model.parameters():
                        param.requires_grad = False
                train_sae = False
       

        ######## BATCH LOOP START ########
        with tqdm(dataloader, unit="batch") as tepoch:
            for batch in tepoch:
                tepoch.set_description(f'{epoch_mode} epoch {epoch}')

                inputs, targets, filename_indices = process_batch(batch, 
                                                                  self.directory_path)
                                                                  #epoch_mode=epoch_mode, 
                                                                  #filename_to_idx_val=self.filename_to_idx_val, 
                                                                  #filename_to_idx_train=self.filename_to_idx_train) 
                inputs, self.targets = inputs.to(self.device), targets.to(self.device)

    def deploy_model(self, num_epochs):
        # if we are evaluating the modified model or the original model, we only perform one epoch
        if not self.training: 
            num_epochs = 1

        if self.use_sae:
            print(f"Using SAE...")
        else:
            print("Using the original model...")

        self.train_batch_idx = 0 # the batch_idx counts the total number of batches used during training across epochs
        self.train_dead_neurons = {}    
        #with torch.autograd.profiler.profile(use_cuda=True) as prof:
        #'''
        for epoch in range(num_epochs):
            if self.training:
                # before the first training epoch we do one evaluation epoch on the test dataset
                if epoch==0: 
                    print("Doing one epoch of evaluation on the test dataset...")
                    #with torch.autograd.profiler.profile(use_cuda=True) as prof1:
                    self.epoch("eval", epoch, num_epochs)
                    #print(prof1.key_averages().table(sort_by="cuda_time_total"))
                
                #with torch.autograd.profiler.profile(use_cuda=True) as prof1:
                print("Doing one epoch of training...")
                self.epoch("train", epoch+1, num_epochs)
                #print(prof1.key_averages().table(sort_by="cuda_time_total"))

            # during evaluation and before every epoch of training we evaluate the model on the validation dataset
            print("Doing one epoch of evaluation on the test dataset...")
            #with torch.autograd.profiler.profile(use_cuda=True) as prof1:
            self.epoch("eval", epoch+1, num_epochs)
            #print(prof1.key_averages().table(sort_by="cuda_time_total"))
        #print(prof.key_averages().table(sort_by="cuda_time_total"))

        print("---------------------------")