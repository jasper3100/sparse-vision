import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange

class SaeMLP(nn.Module):
    def __init__(self, img_size, expansion_factor):
        '''
        Autoencoder used to expand a given intermediate feature map of a model.
        The autoencoder encourages sparsity in this augmented representation and is thus
        called "sparse" autoencoder. See Bricken et al. "Towards Monosemanticity: 
        Decomposing Language Models With Dictionary Learning", 
        https://transformer-circuits.pub/2023/monosemantic-features for more details.

        This SAE has the same architecture as that in the above paper. --> MAYBE NOT TRUE, CHECK

        Parameters
        ----------
        img_size : tuple
            Size of the input image, i.e., (channels, height, width).
        expansion_factor : int
            Factor by which the number of channels is expanded in the encoder.
        '''
        super(SaeMLP, self).__init__()
        self.img_size = img_size
        self.act_size = torch.prod(torch.tensor(self.img_size)).item()
        self.hidden_size = int(self.act_size*expansion_factor)

        self.encoder = nn.Linear(self.act_size, self.hidden_size)
        # we take the transpose of the weight matrix since nn.Linear does x*W^T + b	
        self.encoder.weight = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(self.hidden_size, self.act_size)))
        self.encoder.bias = nn.Parameter(torch.zeros(self.hidden_size))
        self.sae_act = nn.ReLU()
        self.decoder = nn.Linear(self.hidden_size, self.act_size)
        dec_weight = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(self.act_size, self.hidden_size)))
        # We initialize s.t. its columns (rows of the transpose) have unit norm
        # dim=0 --> across the rows --> normalize each column
        # If we consider the tranpose: dim=1 (dim=-1) --> across the columns --> normalize each row
        dec_weight.data[:] = dec_weight / dec_weight.norm(dim=0, keepdim=True)
        self.decoder.weight = dec_weight
        self.decoder.bias = nn.Parameter(torch.zeros(self.act_size))

    def forward(self, x):
        if len(x.shape) == 4:
            x_new = rearrange(x, 'b c h w -> (b h w) c')   
            transformed = True
        else:
            transformed = False
            x_new = x
        x_cent = x_new - self.decoder.bias
        encoder_output_prerelu = self.encoder(x_cent)
        encoder_output = self.sae_act(encoder_output_prerelu)
        decoder_output = self.decoder(encoder_output)
        return encoder_output, decoder_output, encoder_output_prerelu        
        '''
        x_cent = x - self.b_dec
        encoder_output_prerelu = x_cent @ self.W_enc + self.b_enc
        encoder_output = F.relu(encoder_output_prerelu)
        decoder_output = encoder_output @ self.W_dec + self.b_dec 
        return encoder_output, decoder_output, encoder_output_prerelu
        '''

    def make_decoder_weights_and_grad_unit_norm(self):
        # see https://github.com/neelnanda-io/1L-Sparse-Autoencoder/blob/main/utils.py 
        # here we consider the transpose so we use 0 instead of -1
        W_dec = self.decoder.weight
        W_dec_normed = W_dec / W_dec.norm(dim=0, keepdim=True)
        W_dec_grad_proj = (W_dec.grad * W_dec_normed).sum(0, keepdim=True) * W_dec_normed
        W_dec.grad -= W_dec_grad_proj
        W_dec.data = W_dec_normed
        self.decoder.weight = W_dec

    def reset_encoder_weights(self, dead_neurons_sae, device, optimizer, batch_idx):
        W_enc = self.encoder.weight
        b_enc = self.encoder.bias
        W_dec = self.decoder.weight
        b_dec = self.decoder.bias

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
            new_W_enc = (torch.nn.init.kaiming_uniform_(torch.zeros_like(W_enc)))
            new_W_dec = (torch.nn.init.kaiming_uniform_(torch.zeros_like(W_dec)))
            new_b_enc = (torch.zeros_like(b_enc))
            W_enc.data[indices_of_dead_neurons, :] = new_W_enc[indices_of_dead_neurons, :]
            W_dec.data[:, indices_of_dead_neurons] = new_W_dec[:, indices_of_dead_neurons]
            b_enc.data[indices_of_dead_neurons] = new_b_enc[indices_of_dead_neurons]

            self.encoder.weight = W_enc
            self.decoder.weight = W_dec
            self.encoder.bias = b_enc
            
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

                if torch.equal(p, W_enc):
                    # we reset the moving averages for the weights of the dead neurons to zero
                    # optimizer.state[encoder_weight_matrix] -> error, because this matrix has no requires_grad=True
                    # in constrast to p
                    optimizer.state[p]['exp_avg'][indices_of_dead_neurons, :] = torch.zeros_like(W_enc)[indices_of_dead_neurons, :]
                    optimizer.state[p]['exp_avg_sq'][indices_of_dead_neurons, :] = torch.zeros_like(W_enc)[indices_of_dead_neurons, :]
                    # Note that the moving averages might not have changed, i.e., 
                    # (weights_exp_avg_before == optimizer.state[p]['exp_avg']).all() can be True
                    # because the moving averages might have been zero before. This could occur if a neuron was dead from the start.
                    # because the first moving average f.e. is updated as follows:
                    # exp_avg = beta1 * exp_avg + (1 - beta1) * grad --> if a neuron was not active, then grad = 0 and exp_avg = beta1 * exp_avg
                    # but since the initial value is zero, the moving average would be zero
                    # A neuron being dead from the start is particularly likely to happen if we measure dead neurons very early during training, f.e.
                    # after 10 batches.
                else:
                    print("Did not find weight matrix in optimizer params")

                if torch.equal(p, b_enc):
                    #print("Found bias tensor in optimizer params")
                    optimizer.state[p]['exp_avg'][indices_of_dead_neurons] = torch.zeros_like(b_enc)[indices_of_dead_neurons]   
                    optimizer.state[p]['exp_avg_sq'][indices_of_dead_neurons] = torch.zeros_like(b_enc)[indices_of_dead_neurons]  

                if torch.equal(p, W_dec):
                    optimizer.state[p]['exp_avg'][:, indices_of_dead_neurons] = torch.zeros_like(W_dec)[:,indices_of_dead_neurons]
                    optimizer.state[p]['exp_avg_sq'][:, indices_of_dead_neurons] = torch.zeros_like(W_dec)[:,indices_of_dead_neurons]             
            ########## END RESET OPTIMIZER PARAMETERS ##########

            print(f"Batch index {batch_idx}: Re-initialized {len(indices_of_dead_neurons)} dead neurons in the SAE and reset optimizer parameters.")
        else:
            raise ValueError("The indices_of_dead_neurons tensor has unexpected value in second dimension.")