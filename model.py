from torchvision.models import resnet50, ResNet50_Weights

from main import model_name

'''
Define the model to be used.
'''

if model_name == 'resnet50':
    # See https://pytorch.org/vision/stable/models.html for more details on how to use pre-trained models

    # Initialize model with weights
    weights = ResNet50_Weights.IMAGENET1K_V2
    #torch.hub.load("pytorch/vision", "get_weight", weights="ResNet50_Weights.IMAGENET1K_V2")
    model = resnet50(weights=weights)
    #torch.hub.load("pytorch/vision", "resnet50", weights=weights)