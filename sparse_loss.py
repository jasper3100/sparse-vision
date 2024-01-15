import torch
import torch.nn as nn

class SparseLoss(nn.Module):
    '''
    Loss function used to train the sparse autoencoder.
    '''
    def __init__(self, lambda_sparse):
        super(SparseLoss, self).__init__()
        self.lambda_sparse = lambda_sparse

    def forward(self, encoded, decoded, input_data):
        reconstruction_loss = nn.MSELoss()(decoded, input_data)
        # Calculate L1 regularization on hidden layer activations, 
        # i.e. output of encoder, to encourage sparsity
        l1_loss = torch.mean(torch.abs(encoded))
        total_loss = reconstruction_loss + self.lambda_sparse * l1_loss
        return total_loss