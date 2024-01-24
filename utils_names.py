from get_module_names import ModuleNames

def load_module_names(model_name, dataset_name, layer_name=None):
    module_names = ModuleNames(model_name, dataset_name, layer_name)
    return module_names.named_modules()