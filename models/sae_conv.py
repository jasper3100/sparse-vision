import torch.nn as nn

class SaeConv(nn.Module):
    def __init__(self, img_size, expansion_factor):
        '''
        Autoencoder used to expand a given intermediate feature map of a model.
        The autoencoder encourages sparsity in this augmented representation and is thus
        called "sparse" autoencoder. See Bricken et al. "Towards Monosemanticity: 
        Decomposing Language Models With Dictionary Learning", 
        https://transformer-circuits.pub/2023/monosemantic-features for more details.

        Parameters
        ----------
        img_size : tuple
            Size of the input image, i.e., (channels, height, width).
        expansion_factor : int
            Factor by which the number of channels is expanded in the encoder.
        '''
        super(SaeConv, self).__init__()
        # WOULD A BIAS LAYER MAKE SENSE IN OUR CASE??? CAN ONLY DETERMINE THIS ONCE WE KNOW 
        # HOW INTERPRETABLE THE INTERMEDIATE FEATURES ARE I SUPPOSE()
        #self.bias = nn.Parameter(torch.ones(1))

        # the number of channels corresponds to the third last dimension of the input tensor
        # This is invariant to whether there is a batch dimension or not:
        # [batch, channels, height, width] or [channels, height, width]
        #in_channels = input_tensor.size(-3)
        in_channels = img_size[0]
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels*expansion_factor, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        ) 
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels=in_channels*expansion_factor, out_channels=in_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        #x = x + self.bias
        #print(self.bias)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded