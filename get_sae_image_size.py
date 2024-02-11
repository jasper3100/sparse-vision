from utils import *

class GetSaeImgSize:
    '''
    Small class performing inference through the model for one batch
    to store the output dimension of the layer where we want to insert the SAE,
    using hooks. In principle the class structure is similar to that of model pipeline,
    but perfoming this action (get sae img size) there adds too many complications to 
    the workflow.
    '''
    def __init__(self, model, layer_names, train_dataloader):
        self.model = model
        self.layer_names = layer_names
        self.train_dataloader = train_dataloader
        self.hooks = []

    def hook(self, module, input, output, name):
        if name in self.layer_names:
            self.sae_img_size = tuple(output.shape[1:])

    def register_hooks(self):
        module_names = get_module_names(self.model)
        for name in module_names:
            m = getattr(self.model, name)
            hook = m.register_forward_hook(lambda module, inp, out, name=name: self.hook(module, inp, out, name))
            self.hooks.append(hook)

    def get_sae_img_size(self):
        self.register_hooks()

        for batch in self.train_dataloader:
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                inputs, targets = batch
            else:
                raise ValueError("Unexpected data format from dataloader")            
            self.model(inputs)
            break # we only need to do inference through one batch for getting the correct dimension
            
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

        return self.sae_img_size