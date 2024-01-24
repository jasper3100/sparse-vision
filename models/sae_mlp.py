import torch.nn as nn
import torch

class SaeMLP(nn.Module):
    def __init__(self, img_size, expansion_factor):
        '''
        Autoencoder used to expand a given intermediate feature map of a model.
        The autoencoder encourages sparsity in this augmented representation and is thus
        called "sparse" autoencoder. See Bricken et al. "Towards Monosemanticity: 
        Decomposing Language Models With Dictionary Learning", 
        https://transformer-circuits.pub/2023/monosemantic-features for more details.

        This SAE has the same architecture as that in the above paper.

        Parameters
        ----------
        img_size : tuple
            Size of the input image, i.e., (channels, height, width).
        expansion_factor : int
            Factor by which the number of channels is expanded in the encoder.
        '''
        super(SaeMLP, self).__init__()
        self.img_size = img_size
        self.prod_size = torch.prod(torch.tensor(self.img_size)).item()

        # WOULD A BIAS LAYER MAKE SENSE IN OUR CASE??? CAN ONLY DETERMINE THIS ONCE WE KNOW 
        # HOW INTERPRETABLE THE INTERMEDIATE FEATURES ARE I SUPPOSE()
        self.bias = nn.Parameter(torch.ones(1))
        self.encoder = nn.Sequential(
            nn.Linear(self.prod_size, self.prod_size*expansion_factor, bias=True),
            nn.ReLU()
        )
        self.decoder = nn.Linear(self.prod_size*expansion_factor, self.prod_size, bias=True)

    def forward(self, x):
        x = x + self.bias
        #print(self.bias)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded