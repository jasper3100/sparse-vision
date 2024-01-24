from torchvision.models import resnet50, ResNet50_Weights

from models.custom_mlp_1 import CustomMLP1
from models.sae_conv import SaeConv
from models.sae_mlp import SaeMLP

class ModelLoader:
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = None

    def load_model(self, img_size=None, expansion_factor=None):
        if self.model_name == 'resnet50':
            # Initialize model with weights
            self.weights = ResNet50_Weights.IMAGENET1K_V2
            self.model = resnet50(weights=self.weights)
            self.model.eval() # model is pre-trained and we don't train it
            # Alternatively could try:
            # torch.hub.load("pytorch/vision", "get_weight", weights="ResNet50_Weights.IMAGENET1K_V2")
            # torch.hub.load("pytorch/vision", "resnet50", weights=weights)
        elif self.model_name == 'custom_mlp_1':
            self.model = CustomMLP1(img_size)
            self.weights = None
        elif self.model_name == 'sae_conv':
            self.model = SaeConv(img_size, expansion_factor)
            self.weights = None
        elif self.model_name == 'sae_mlp':
            self.model = SaeMLP(img_size, expansion_factor)
            self.weights = None
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")

'''
if __name__ == '__main__':
    # Example usage
    model_loader = ModelLoader(model_name)
    model_loader.load_model()

    # Access the loaded model
    model = model_loader.model
    # access the model weights
    weights = model.weights
'''