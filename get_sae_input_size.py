from utils import *

class GetSaeInpSize:
    '''
    Small class performing inference through the model for one batch
    to store the output dimension of the layer where we want to insert the SAE,
    using hooks. In principle the class structure is similar to that of model pipeline,
    but perfoming this action (get sae input size) there adds too many complications to 
    the workflow.
    '''
    def __init__(self, model, layer_name, train_dataloader, device, model_name):
        self.model = model
        self.layer_name = layer_name
        self.train_dataloader = train_dataloader
        self.hooks = []
        self.device = device
        self.model_name = model_name

    def hook(self, module, input, output, name):
        if name == self.layer_name:
            self.sae_inp_size = tuple(output.shape)
            # if the inp size has 4 dimensions, we assume for now that this is the output of a conv layer
            # for the output of a conv layer, we do (BS, C, H, W) → (BS*W*H, C), we consider BS*W*H as the new batch size and
            # each channel as a unit that we wish to interpret, i.e., analogous to the case of linear layers we return the 
            # number of channels
            if len(self.sae_inp_size) == 4:
                self.sae_inp_size = self.sae_inp_size[1]
            elif len(self.sae_inp_size) == 2: # we assume this to be the output of a linear layer (BS, number_of_neurons)
                self.sae_inp_size = self.sae_inp_size[1:]
            else:
                raise ValueError("Unexpected output shape from layer")
        
    def register_hooks(self):
        # we register a hook on the layer whose size we want to get
        m = None
        # separate name based on "."
        for subname in self.layer_name.split("."):
            if m is None:
                m = getattr(self.model, subname)
            else: 
                m = getattr(m, subname)
        # for example, for 'layer1.0.conv1' -->  m = getattr(model, 'layer1') --> m = getattr(m, '0') --> m = getattr(m, 'conv1')
        hook = m.register_forward_hook(lambda module, inp, out, name=self.layer_name: self.hook(module, inp, out, name))
        # manually: hook = model.layer1[0].conv1.register_forward_hook(lambda module, inp, out, name='layer1.0.conv1', use_sae=use_sae, train_sae=train_sae: self.hook(module, inp, out, name, use_sae, train_sae))
        self.hooks.append(hook)

    def get_sae_inp_size(self):
        self.register_hooks()
        
        for batch in self.train_dataloader:
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                inputs, targets = batch
            elif isinstance(batch, dict) and len(batch) == 2 and list(batch.keys())[0] == "image" and list(batch.keys())[1] == "label":
                # this format holds for the tiny imagenet dataset
                inputs, targets = batch["image"], batch["label"]
            else:
                raise ValueError("Unexpected data format from dataloader")  
            # get the first sample from the batch
            inputs = inputs[0].unsqueeze(0)   
            inputs = inputs.to(self.device)       
            self.model(inputs)
            break # we only need to do inference through one batch/sample for getting the correct dimension
            
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

        return self.sae_inp_size