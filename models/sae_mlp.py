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

        self.bias = nn.Parameter(torch.ones(self.prod_size))
        self.encoder = nn.Sequential(
            nn.Linear(self.prod_size, int(self.prod_size*expansion_factor), bias=True),
            nn.ReLU()
        )
        self.decoder = nn.Linear(int(self.prod_size*expansion_factor), self.prod_size, bias=True)

    def forward(self, x):
        x = x + self.bias
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

    def reset_encoder_weights(self, dead_neurons_sae, device, optimizer, batch_idx):
        encoder_weight_matrix = self.encoder[0].weight.data # we do [0] because the linear layer is the first layer in the encoder Sequential
        encoder_bias_tensor = self.encoder[0].bias.data
        decoder_weight_matrix = self.decoder.weight.data # no need to do [0] because there is only one layer in the decoder

        indices_of_dead_neurons = torch.nonzero(dead_neurons_sae)
        print(indices_of_dead_neurons)
        print(indices_of_dead_neurons.shape)
        # --> shape (n,1) where n>=0 is number of dead neurons 
        if len(indices_of_dead_neurons.shape) != 2:
            raise ValueError(f"Batch index {batch_idx}: The indices_of_dead_neurons tensor has unexpected shape.")
        
        if indices_of_dead_neurons.shape[0] == 0 or indices_of_dead_neurons.shape[1] == 0:
            # there are no dead neurons so we don't need to re-initialize anything
            print(f"Batch index {batch_idx}: No dead neurons in the SAE --> no re-initialization necessary")

        elif indices_of_dead_neurons.shape[1] == 1:
            indices_of_dead_neurons = torch.squeeze(indices_of_dead_neurons, dim=-1)
            # --> shape (n)
            # Reset the encoding weights and biases leading to the dead neurons and those going out of them
            # in the decoder, using He/Kaiming initialization
            # a torch weight matrix has dimension [out_features, in_features] --> our encoder weight matrix should have shape [len(indices), in_features = self.prod_size]
            encoder_weights_to_be_reinitialized = encoder_weight_matrix[indices_of_dead_neurons,:]
            # --> shape (n, self.prod_size) where n is the number of dead neurons, n = len(indices), we verify that the shape is correct
            #print(encoder_weights_to_be_adjusted.shape)
            if encoder_weights_to_be_reinitialized.shape != (len(indices_of_dead_neurons), self.prod_size):
                raise ValueError(f'The matrix of the encoder weights to be re-initialized has shape {encoder_weights_to_be_reinitialized.shape} which is unexpected.')
            encoder_biases_to_be_reinitialized = encoder_bias_tensor[indices_of_dead_neurons]

            decoder_weights_to_be_reinitialized = decoder_weight_matrix[:,indices_of_dead_neurons]
            if decoder_weights_to_be_reinitialized.shape != (self.prod_size, len(indices_of_dead_neurons)):
                raise ValueError(f'The matrix of the decoder weights to be re-initialized has shape {decoder_weights_to_be_reinitialized.shape} which is unexpected.')

            ########## START RESET WEIGHTS AND BIASES ##########
            encoder_weight_matrix_before = encoder_weight_matrix.clone()
            encoder_bias_tensor_before = encoder_bias_tensor.clone()
            decoder_weight_matrix_before = decoder_weight_matrix.clone()
            encoder_weight_matrix[indices_of_dead_neurons,:] = nn.init.kaiming_uniform_(encoder_weights_to_be_reinitialized, mode='fan_in', nonlinearity='relu')
            # doing nn.init.kaiming_uniform_(...) directly without assigning it to the matrix only works 
            # if we take the full matrix, not a slice of it
            encoder_bias_tensor[indices_of_dead_neurons] = nn.init.zeros_(encoder_biases_to_be_reinitialized)
            decoder_weight_matrix[:,indices_of_dead_neurons] = nn.init.kaiming_uniform_(decoder_weights_to_be_reinitialized, mode='fan_in', nonlinearity='relu')

            # verify that the weights and biases were changed
            if (encoder_weight_matrix_before == encoder_weight_matrix).all():
                raise ValueError("The encoder weights were not changed.")
            if (encoder_bias_tensor_before == encoder_bias_tensor).all():
                raise ValueError("The encoder biases were not changed.")
            if (decoder_weight_matrix_before == decoder_weight_matrix).all():
                raise ValueError("The decoder weights were not changed.")
            ########## END RESET WEIGHTS AND BIASES ##########
            
            ########## START RESET OPTIMIZER PARAMETERS ##########
            # Reset the optimizer state for the specified indices; Adam --> reset the moving averages
            if optimizer.__class__.__name__ != 'Adam':
                raise ValueError(f"The optimizer {optimizer.__class__.__name__} is not supported for re-initializing dead neurons.")
            
            # what do we do about step of Adam??? because step matters and should not be universal acrosss all weigths...

            for p in optimizer.param_groups[0]['params']:
                #print("Shape of Adam param", p.shape)
                # For a layer in the base model of size [256] and expansion_factor = 2, we get: 
                #[1] (bias term "before" encoder), [512, 256] (weight matrix in encoder), [256] (bias term in decoder), 
                #[256, 512] (weight matrix in decoder), [256] (bias term in decoder)

                if torch.equal(p, encoder_weight_matrix):
                    #print("Found weight matrix in optimizer params")
                    #weights_exp_avg_before = optimizer.state[p]['exp_avg'].clone()
                    #weights_exp_avg_sq_before = optimizer.state[p]['exp_avg_sq'].clone()
                    # we reset the moving averages for the weights of the dead neurons to zero
                    # optimizer.state[encoder_weight_matrix] -> error, because this matrix has no requires_grad=True
                    # in constrast to p
                    optimizer.state[p]['exp_avg'][indices_of_dead_neurons,:] = torch.zeros_like(encoder_weights_to_be_reinitialized)
                    optimizer.state[p]['exp_avg_sq'][indices_of_dead_neurons,:] = torch.zeros_like(encoder_weights_to_be_reinitialized)
                    # Note that the moving averages might not have changed, i.e., 
                    # (weights_exp_avg_before == optimizer.state[p]['exp_avg']).all() can be True
                    # because the moving averages might have been zero before. This could occur if a neuron was dead from the start.
                    # because the first moving average f.e. is updated as follows:
                    # exp_avg = beta1 * exp_avg + (1 - beta1) * grad --> if a neuron was not active, then grad = 0 and exp_avg = beta1 * exp_avg
                    # but since the initial value is zero, the moving average would be zero
                    # A neuron being dead from the start is particularly likely to happen if we measure dead neurons very early during training, f.e.
                    # after 10 batches.

                if torch.equal(p, encoder_bias_tensor):
                    #print("Found bias tensor in optimizer params")
                    optimizer.state[p]['exp_avg'][indices_of_dead_neurons] = torch.zeros_like(encoder_biases_to_be_reinitialized)   
                    optimizer.state[p]['exp_avg_sq'][indices_of_dead_neurons] = torch.zeros_like(encoder_biases_to_be_reinitialized)  

                if torch.equal(p, decoder_weight_matrix):
                    optimizer.state[p]['exp_avg'][:,indices_of_dead_neurons] = torch.zeros_like(decoder_weights_to_be_reinitialized)
                    optimizer.state[p]['exp_avg_sq'][:,indices_of_dead_neurons] = torch.zeros_like(decoder_weights_to_be_reinitialized)             
            ########## END RESET OPTIMIZER PARAMETERS ##########

            print(f"Batch index {batch_idx}: Re-initialized {len(indices_of_dead_neurons)} dead neurons in the SAE and reset optimizer parameters.")
        else:
            raise ValueError("The indices_of_dead_neurons tensor has unexpected value in second dimension.")