from utils import *

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
        # nn.Linear does x*W^T + b --> x has dim. (Batch dim, act_size), output has shape (batch dim, hidden_size)
        # --> W^T has shape (act_size, hidden_size) --> W (encoder weight) has shape (hidden_size, act_size)
        self.encoder.weight = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(self.hidden_size, self.act_size)))
        self.encoder.bias = nn.Parameter(torch.zeros(self.hidden_size))
        self.sae_act = nn.ReLU()
        self.decoder = nn.Linear(self.hidden_size, self.act_size)
        self.decoder.bias = nn.Parameter(torch.zeros(self.act_size))
        # dec_weight has shape (act_size, hidden_size) by same argument as above but all quantities reversed
        dec_weight = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(self.act_size, self.hidden_size)))
        # We initialize s.t. its columns (rows of the transpose) have unit norm
        # dim=0 --> across the rows --> normalize each column
        # If we consider the tranpose: dim=1 (dim=-1) --> across the columns --> normalize each row
        dec_weight.data[:] = dec_weight / dec_weight.norm(dim=0, keepdim=True)
        self.decoder.weight = dec_weight

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

    '''
    def make_decoder_weights_and_grad_unit_norm(self):
        # WARNING THIS FUNCTION INCREASES MEMORY USAGE BY ATTACHING ADDITIONAL GRADIENTS!!!
        # Because: grad = grad - grad_proj --> next batch: (grad - grad_proj) - grad_proj etc.
        # so we always get one more grad_proj --> memory increase; grad_proj is always different
        # because W_dec is updated in between
        # USE CONSTRAINED ADAM OPTIMIZER INSTEAD!
        # see https://github.com/neelnanda-io/1L-Sparse-Autoencoder/blob/main/utils.py 
        # here we consider the transpose so we use 0 instead of -1
        W_dec = self.decoder.weight
        W_dec_normed = W_dec / W_dec.norm(dim=0, keepdim=True)
        W_dec_grad_proj = (W_dec.grad * W_dec_normed).sum(0, keepdim=True) * W_dec_normed
        W_dec.grad -= W_dec_grad_proj
        W_dec.data = W_dec_normed
        #self.decoder.weight = W_dec
    '''

    def reset_encoder_weights(self, dead_neurons_sae, device, optimizer, epoch, train_batch_idx, epoch_batch_idx, file_path):
        W_enc = self.encoder.weight
        b_enc = self.encoder.bias
        W_dec = self.decoder.weight
        #print(W_dec.grad) --> None

        indices_of_dead_neurons = torch.nonzero(dead_neurons_sae)
        #print(indices_of_dead_neurons)
        #print(indices_of_dead_neurons.shape)
        # --> shape (n,1) where n>=0 is number of dead neurons 
        if len(indices_of_dead_neurons.shape) != 2:
            raise ValueError(f"Epoch {epoch}, train batch index {train_batch_idx}, epoch batch index {epoch_batch_idx}: The indices_of_dead_neurons tensor has unexpected shape.")
        
        elif indices_of_dead_neurons.shape[0] == 0 or indices_of_dead_neurons.shape[1] == 0:
            # there are no dead neurons so we don't need to re-initialize anything
            print(f"Epoch {epoch}, train batch index {train_batch_idx}, epoch batch index {epoch_batch_idx}: No dead neurons in the SAE --> no re-initialization necessary")

        elif indices_of_dead_neurons.shape[1] == 1:
            indices_of_dead_neurons = torch.squeeze(indices_of_dead_neurons, dim=-1)
            # --> shape (n)
                        
            with open(file_path, 'w') as file:
                for element in indices_of_dead_neurons: # save each element on a new line
                    file.write(str(element.item()) + '\n')

            # Reset the encoding weights and biases leading to the dead neurons and those going out of them
            # in the decoder, using He/Kaiming initialization
            new_W_enc = (torch.nn.init.kaiming_uniform_(torch.zeros_like(W_enc)))
            new_W_dec = (torch.nn.init.kaiming_uniform_(torch.zeros_like(W_dec)))
            new_b_enc = (torch.zeros_like(b_enc))

            # compute the average L2 norm of the weights in the non-dead neurons to scale the new weights accordingly
            # in particular, we take the average L2 norm of all weights going into non-dead SAE features and those going out of them
            # because those are the weights we re-initialize --> calculate L2 over the input dimension of a layer (i.e. dimension of model layer) -> dim=1
            indices_of_non_dead_neurons = torch.nonzero(~dead_neurons_sae)
            indices_of_non_dead_neurons = torch.squeeze(indices_of_non_dead_neurons, dim=-1)
            L2_W_enc = torch.norm(W_enc[indices_of_non_dead_neurons, :], p=2, dim=1) # dim: [hidden_size], i.e. we have the average
            # L2 norm of the weights going into each SAE feature
            average_L2_W_enc = torch.mean(L2_W_enc).item()
            L2_W_dec = torch.norm(W_dec[:, indices_of_non_dead_neurons], p=2, dim=1) # dim: [act_size] --> should be [hidden_size] no???
            average_L2_W_dec = torch.mean(L2_W_dec).item()
            # for the bias, for each entry, the L2 norm is just the absolute value
            L2_b_enc = torch.mean(torch.abs(b_enc[indices_of_non_dead_neurons])).item()

            #'''
            # Normalize each row to have the desired L2 norm
            W_enc_row_norms = torch.norm(new_W_enc, p=2, dim=1, keepdim=True)  # Calculate L2 norm of each row
            new_W_enc = new_W_enc / W_enc_row_norms * average_L2_W_enc  # Normalize each row
            W_dec_row_norms = torch.norm(new_W_dec, p=2, dim=1, keepdim=True)  # Calculate L2 norm of each row
            new_W_dec = new_W_dec / W_dec_row_norms * average_L2_W_dec  # Normalize each row
            # we set each bias value to the average L2 norm of the bias of the non-dead neurons
            new_b_enc = torch.full_like(new_b_enc, L2_b_enc) 
            #'''

            W_enc.data[indices_of_dead_neurons, :] = new_W_enc[indices_of_dead_neurons, :]
            W_dec.data[:, indices_of_dead_neurons] = new_W_dec[:, indices_of_dead_neurons]
            b_enc.data[indices_of_dead_neurons] = new_b_enc[indices_of_dead_neurons]

            # we make sure that the decoder weights have norm one as in the first initialization
            W_dec.data[:] = W_dec.data / W_dec.data.norm(dim=0, keepdim=True) 
            # W_dec has requires_grad=True, W_dec.data doesn't have it...
            
            ########## START RESET OPTIMIZER PARAMETERS ##########
            # Reset the optimizer state for the specified indices; Adam --> reset the moving averages
            if optimizer.__class__.__name__ != 'Adam' and optimizer.__class__.__name__ != 'ConstrainedAdam':
                raise ValueError(f"The optimizer {optimizer.__class__.__name__} is not supported for re-initializing dead neurons.")
            
            # what do we do about step of Adam??? because step matters and should not be universal acrosss all weigths...

            for p in optimizer.param_groups[0]['params']:
                #print("Shape of Adam param", p.shape)
                # For a layer in the base model of size [256] and expansion_factor = 2, we get: 
                #[1] (bias term "before" encoder), [512, 256] (weight matrix in encoder), [256] (bias term in decoder), 
                #[256, 512] (weight matrix in decoder), [256] (bias term in decoder)

                if torch.equal(p, W_enc):
                    print("Found encoder weight tensor in optimizer params")
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
                elif torch.equal(p, b_enc):
                    print("Found encoder bias tensor in optimizer params")
                    optimizer.state[p]['exp_avg'][indices_of_dead_neurons] = torch.zeros_like(b_enc)[indices_of_dead_neurons]   
                    optimizer.state[p]['exp_avg_sq'][indices_of_dead_neurons] = torch.zeros_like(b_enc)[indices_of_dead_neurons]  
                elif torch.equal(p, W_dec):
                    print("Found decoder weight tensor in optimizer params")
                    optimizer.state[p]['exp_avg'][:, indices_of_dead_neurons] = torch.zeros_like(W_dec)[:,indices_of_dead_neurons]
                    optimizer.state[p]['exp_avg_sq'][:, indices_of_dead_neurons] = torch.zeros_like(W_dec)[:,indices_of_dead_neurons]  
                #else:
                    #print("Optimizer param doesn't correspond to encoder weight, encoder bias or decoder weight.")
           
            ########## END RESET OPTIMIZER PARAMETERS ##########

            print(f"Epoch {epoch}, train batch index {train_batch_idx}, epoch batch index {epoch_batch_idx}: Re-initialized {len(indices_of_dead_neurons)} dead neurons in the SAE and reset optimizer parameters.")
        else:
            raise ValueError(f"Epoch {epoch}, train batch index {train_batch_idx}, epoch batch index {epoch_batch_idx}: The indices_of_dead_neurons tensor has unexpected value in second dimension.")


    def intervene_on_decoder_weights(self, unit_index, value):
        '''
        unit index: int, index of the SAE unit/feature to intervene on
        value: float, value to set the unit to (in the case of imagenet, this can be the imagenet mean)
        '''
        # W_dec has shape (self.act_size, self.hidden_size), where self.act_size is the size of the incoming activation
        # f.e. self.act_size = 64 and self.hidden_size = 128 (exp. fact. 2)
        # the decoder is nn.Linear, which does x*W^T + b
        # W_dec^T has shape (self.hidden_size, self.act_size), f.e. (128, 64) (and x has shape (1,128))
        # --> each row of W_dec^T is a feature of the SAE, i.e., there are 128 features (of size 64 each), which makes sense
        # --> each column of W_dec is a feature of the SAE
        # Hence, we set the unit_index-th column of W_dec to the Imagenet mean (or we could take the mean of the image cluster that we consider)
        self.decoder.weight.data[:, unit_index] = value

        # make sure to normalize the decoder weight matrix with ""