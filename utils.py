import torch.nn.functional as F

from model_loader import ModelLoader
from input_data_loader import InputDataLoader

def load_model_aux(model_name, img_size, expansion_factor=None):
    model_loader = ModelLoader(model_name)
    model_loader.load_model(img_size, expansion_factor)
    return model_loader.model, model_loader.weights

def load_data_aux(dataset_name, batch_size, data_dir=None, layer_name=None):
    data_loader = InputDataLoader(dataset_name, batch_size)
    data_loader.load_data(root_dir=data_dir, layer_name=layer_name)
    return data_loader.train_data, data_loader.val_data, data_loader.img_size, data_loader.category_names

def compute_ce(feature_map_1, feature_map_2):
    # cross_entropy(input, target), where target consists of probabilities
    feature_map_1 = F.softmax(feature_map_1, dim=1)
    #feature_map_1 = feature_map_1.to(torch.float64)
    #feature_map_2 = feature_map_2.to(torch.float64)
    return F.cross_entropy(feature_map_2, feature_map_1)