import torch.nn as nn
import torch.nn.functional as F
import torch

'''
IT IS IMPORTANT THAT EACH LAYER IS ONLY USED ONCE IN THE MODEL. 
For instance, if we do: self.act = nn.ReLU() and then in the forward
method: x = self.act(self.fc1(x)) and x = self.act(self.fc2(x)), then
the activations of the act layer will contain activations from 2 different
places, which will lead to confusions!
'''

class CustomMLP1(nn.Module):
    def __init__(self, img_size, num_classes=None):
        # call the constructor of the parent class
        super(CustomMLP1, self).__init__()
        self.img_size = img_size
        self.num_classes = num_classes if num_classes is not None else 10
        self.prod_size = torch.prod(torch.tensor(self.img_size)).item()

        # define the layers
        self.fc1 = nn.Linear(self.prod_size, 256)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 256)
        self.act2 = nn.ReLU()
        self.fc3 = nn.Linear(256, self.num_classes)
        #self.sm = nn.Softmax(dim=1)

    def forward(self, x):
        # flatten the input
        x = x.view(-1, self.prod_size)
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        #x = self.sm(self.fc3(x))
        # don't use softmax as last layer when using CrossEntropyLoss as loss
        # function because it expects unnormalized input
        x = self.fc3(x)
        return x
    
class CustomMLP2(nn.Module):
    def __init__(self, img_size, num_classes=None):
        super(CustomMLP2, self).__init__()
        self.img_size = img_size
        self.num_classes = num_classes if num_classes is not None else 10
        self.prod_size = torch.prod(torch.tensor(self.img_size)).item()

        # Define the layers
        self.fc1 = nn.Linear(self.prod_size, 1024)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(1024, 512)
        self.act2 = nn.ReLU()
        self.fc3 = nn.Linear(512, 256)
        self.act3 = nn.ReLU()
        self.fc4 = nn.Linear(256, 128)
        self.act4 = nn.ReLU()
        self.fc5 = nn.Linear(128, self.num_classes)

    def forward(self, x):
        x = x.view(-1, self.prod_size)
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        x = self.act3(self.fc3(x))
        x = self.act4(self.fc4(x))
        x = self.fc5(x)
        return x
    

class CustomMLP3(nn.Module):
    def __init__(self, img_size, num_classes=None):
        super(CustomMLP3, self).__init__()
        self.img_size = img_size
        self.num_classes = num_classes if num_classes is not None else 10
        self.prod_size = torch.prod(torch.tensor(self.img_size)).item()

        # Define the layers
        self.fc1 = nn.Linear(self.prod_size, 64)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.act2 = nn.ReLU()
        self.fc3 = nn.Linear(32, 16)
        self.act3 = nn.ReLU()
        self.fc4 = nn.Linear(16, self.num_classes)

    def forward(self, x):
        x = x.view(-1, self.prod_size)
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        x = self.act3(self.fc3(x))
        x = self.fc4(x)
        return x
    
class CustomMLP4(nn.Module):
    def __init__(self, img_size, num_classes=None):
        super(CustomMLP4, self).__init__()
        self.img_size = img_size
        self.num_classes = num_classes if num_classes is not None else 10
        self.prod_size = torch.prod(torch.tensor(self.img_size)).item()

        # Define the layers
        self.fc1 = nn.Linear(self.prod_size, 32)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(32, 16)
        self.act2 = nn.ReLU()
        self.fc3 = nn.Linear(16, 16)
        self.act3 = nn.ReLU()
        self.fc4 = nn.Linear(16, self.num_classes)

    def forward(self, x):
        x = x.view(-1, self.prod_size)
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        x = self.act3(self.fc3(x))
        x = self.fc4(x)
        return x

class CustomMLP5(nn.Module):
    def __init__(self, img_size, num_classes=None):
        super(CustomMLP5, self).__init__()
        self.img_size = img_size
        self.num_classes = num_classes if num_classes is not None else 10
        self.prod_size = torch.prod(torch.tensor(self.img_size)).item()

        # Define the layers
        self.fc1 = nn.Linear(self.prod_size, 10)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(10, 10)
        self.act2 = nn.ReLU()
        self.fc3 = nn.Linear(10, 10)
        self.act3 = nn.ReLU()
        self.fc4 = nn.Linear(10, self.num_classes)

    def forward(self, x):
        x = x.view(-1, self.prod_size)
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        x = self.act3(self.fc3(x))
        x = self.fc4(x)
        return x
    
class CustomMLP6(nn.Module):
    def __init__(self, img_size, num_classes=None):
        super(CustomMLP6, self).__init__()
        self.img_size = img_size
        self.num_classes = num_classes if num_classes is not None else 10
        self.prod_size = torch.prod(torch.tensor(self.img_size)).item()

        # Define the layers
        self.fc1 = nn.Linear(self.prod_size, 64)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.act2 = nn.ReLU()
        self.fc3 = nn.Linear(32, 5)
        self.act3 = nn.ReLU()
        self.fc4 = nn.Linear(5, 16)
        self.act4 = nn.ReLU()
        self.fc5 = nn.Linear(16, self.num_classes)

    def forward(self, x):
        x = x.view(-1, self.prod_size)
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        x = self.act3(self.fc3(x))
        x = self.act4(self.fc4(x))
        x = self.fc5(x)
        return x
    
class CustomMLP7(nn.Module):
    # same as MLP4, but without the activation functions
    def __init__(self, img_size, num_classes=None):
        super(CustomMLP7, self).__init__()
        self.img_size = img_size
        self.num_classes = num_classes if num_classes is not None else 10
        self.prod_size = torch.prod(torch.tensor(self.img_size)).item()

        # Define the layers
        self.fc1 = nn.Linear(self.prod_size, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 16)
        self.fc4 = nn.Linear(16, self.num_classes)

    def forward(self, x):
        x = x.view(-1, self.prod_size)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x
    
class CustomMLP8(nn.Module):
    # same as MLP4, but with one layer less
    def __init__(self, img_size, num_classes=None):
        super(CustomMLP8, self).__init__()
        self.img_size = img_size
        self.num_classes = num_classes if num_classes is not None else 10
        self.prod_size = torch.prod(torch.tensor(self.img_size)).item()

        # Define the layers
        self.fc1 = nn.Linear(self.prod_size, 32)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(32, 16)
        self.act2 = nn.ReLU()
        self.fc3 = nn.Linear(16, self.num_classes)

    def forward(self, x):
        x = x.view(-1, self.prod_size)
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        x = self.fc3(x)
        return x

class CustomMLP9(nn.Module):
    # same as MLP8, but with fewer neurons per layer, and with one layer less
    def __init__(self, img_size, num_classes=None):
        super(CustomMLP9, self).__init__()
        self.img_size = img_size
        self.num_classes = num_classes if num_classes is not None else 10
        self.prod_size = torch.prod(torch.tensor(self.img_size)).item()

        # Define the layers
        self.fc1 = nn.Linear(self.prod_size, 16)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(16, self.num_classes)

    def forward(self, x):
        x = x.view(-1, self.prod_size)
        x = self.act1(self.fc1(x))
        x = self.fc2(x)
        return x
    
class CustomMLP9_SAE_fc1(nn.Module):
    # MLP9 but with SAE MPL inserted after fc1
    def __init__(self, img_size, expansion_factor, W_enc, W_dec, b_enc, b_dec, num_classes=None):
        super(CustomMLP9_SAE_fc1, self).__init__()
        self.img_size = img_size
        self.num_classes = num_classes if num_classes is not None else 10
        self.prod_size = torch.prod(torch.tensor(self.img_size)).item()

        # We define the weights of the SAE
        act_size = 16
        hidden_size = int(act_size*expansion_factor)

        # Define the layers
        self.fc1 = nn.Linear(self.prod_size, 16)
        self.encoder = nn.Linear(16, hidden_size)
        self.encoder.weight = nn.Parameter(W_enc.t()) # we take the transpose here because nn.Linear does x*weight.T + b_enc 
        self.encoder.bias = nn.Parameter(b_enc)
        self.encoder_act = nn.ReLU()
        self.decoder = nn.Linear(hidden_size, 16)
        self.decoder.weight = nn.Parameter(W_dec.t())
        self.decoder.bias = nn.Parameter(b_dec)
        # in the SAE we also want the rows of W_dec to have unit norm but since we will only use this
        # model with pre-trained weights anyways, we don't need to worry about this here
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(16, self.num_classes)
    
    def forward(self, x):
        x = x.view(-1, self.prod_size)
        x = self.fc1(x)
        # insert SAE
        x_cent = x - self.decoder.bias
        encoder_output = self.encoder_act(self.encoder(x_cent)) # should be equivalent to: F.relu(x_cent @ self.W_enc + self.b_enc)
        decoder_output = self.decoder(encoder_output) # should be equivalent to: encoder_output @ self.W_dec + self.b_dec 
        # continue with base model
        x = self.act1(decoder_output)
        x = self.fc2(x)
        return x
    
    '''# use it in model_pipeline.py like this:
    model_with_sae = CustomMLP9_SAE_fc1(self.img_size, 
                                        self.sae_expansion_factor, 
                                        self.model_sae_fc1.W_enc, 
                                        self.model_sae_fc1.W_dec, 
                                        self.model_sae_fc1.b_enc, 
                                        self.model_sae_fc1.b_dec, 
                                        self.num_classes) 
    model_with_sae = model_with_sae.to(self.device)
    model_with_sae.fc1.weight.data = self.model.fc1.weight.data
    model_with_sae.fc1.bias.data = self.model.fc1.bias.data
    model_with_sae.fc2.weight.data = self.model.fc2.weight.data
    model_with_sae.fc2.bias.data = self.model.fc2.bias.data
    '''
    
class CustomMLP10(nn.Module):
    # same as MLP9, but with fewer neurons per layer
    def __init__(self, img_size, num_classes=None):
        super(CustomMLP10, self).__init__()
        self.img_size = img_size
        self.num_classes = num_classes if num_classes is not None else 10
        self.prod_size = torch.prod(torch.tensor(self.img_size)).item()

        # Define the layers
        self.fc1 = nn.Linear(self.prod_size, 10)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(10, self.num_classes)

    def forward(self, x):
        x = x.view(-1, self.prod_size)
        x = self.act1(self.fc1(x))
        x = self.fc2(x)
        return x