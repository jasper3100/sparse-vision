import torch
import torch.nn as nn

class SparseLoss(nn.Module):
    '''
    Loss function used to train the sparse autoencoder.
    '''
    def __init__(self, lambda_sparse):
        super(SparseLoss, self).__init__()
        self.lambda_sparse = lambda_sparse

    def forward(self, encoded, decoded, targets):
        reconstruction_loss = nn.MSELoss()(decoded, targets)
        # Calculate L1 regularization on hidden layer activations, 
        # i.e. output of encoder, to encourage sparsity
        l1_loss = torch.mean(torch.abs(encoded))
        return reconstruction_loss, l1_loss
    
'''
#Alternatively (is this better?)
def make_sparse_loss(lambda_sparse):
    def sparse_loss(outputs, targets):
        # targets will be the input data
        encoded, decoded = outputs
        reconstruction_loss = nn.MSELoss()(decoded, targets)
        # Calculate L1 regularization on hidden layer activations, 
        # i.e. output of encoder, to encourage sparsity
        l1_loss = torch.mean(torch.abs(encoded))
        total_loss = reconstruction_loss + lambda_sparse * l1_loss
        return total_loss
    return sparse_loss

fixed_sparse_loss = make_sparse_loss(lambda_sparse=0.1)

# Example usage
outputs = (torch.randn(10, 10), torch.randn(10, 10))
targets = torch.randn(10, 10)

loss_value = fixed_sparse_loss(outputs, targets)
print(loss_value)
'''