import re

from utils import load_model

class ModuleNames:
    def __init__(self, model_name):
        self.model_name = model_name
        self.model, _ = load_model(self.model_name)

    def convert_name(self, name):
        """
        Convert .1.2.3. to [1][2][3]. and add "model." to the beginning
        """
        name_brackets = re.sub(r'\.(\d+)', r'[\1]', name)
        return f"model.{name_brackets}"

    def get_lowest_level_modules(self):
        """
        Get all model layers (layers at the lowest level in case of nested submodules)
        and their names. Getting the names can be useful for printing all layers to 
        see which ones are available for choosing as the specific layer where to apply the SAE.
        """
        layers = {}
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:
                # do not consider empty strings if any
                if name:
                    name = self.convert_name(name)
                    layers[name] = module
        
        ''' # as a list of tuples
        layers = []
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:
                layers.append((name, module))
        # remove empty strings if any
        layers = [(name, module) for name, module in layers if name]
        # convert the format of the names
        layers = [(self.convert_name(name), module) for name, module in layers]
        '''
        return layers

    def get_main_modules_and_names(self):
        """
        Get names and modules of the main modules of the model.
        """
        names = []
        modules = []
        for name, module in self.model.named_children():
            names.append(name)
            modules.append(module)
        return names, modules

    def names_of_main_modules_and_specified_layer(self, layer_name):
        """
        Get the names of the main modules of the model and append the specified layer_name.
        """
        module_names, _ = self.get_main_modules_and_names()
        module_names = [f"model.{name}" for name in module_names]
        module_names.append(layer_name)
        return module_names

#'''
# Example usage:
'''
model_name = 'resnet50'
module_names = ModuleNames(model_name)
layers = module_names.get_lowest_level_modules()
#print(layers.keys())
#print(layers.values())
module = list(layers.values())[-1]
#module.register_forward_hook(lambda module, inp, out, name=name: self.hook(module, inp, out, name))

model, _ = load_model(model_name)
layer = model._modules.get("layer1[0].conv1")
print(layer)
'''

#names, modules = module_names.get_main_modules_and_names()
#print(modules)

#specified_layer_names = module_names.names_of_main_modules_and_specified_layer('model.layer1[0].conv3')
#print(specified_layer_names)
#'''
