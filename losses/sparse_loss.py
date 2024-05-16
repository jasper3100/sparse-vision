import torch
import torch.nn as nn

class SparseLoss(nn.Module):
    '''
    Loss function used to train the sparse autoencoder.
    '''
    def __init__(self):
        super(SparseLoss, self).__init__()

    def forward(self, encoded, decoded, targets):
        '''
        the input arguments are batches of data, we want to calculate the loss for each 
        sample in the batch and then average the loss over the batch
        '''
        reconstruction_loss = nn.MSELoss()(decoded, targets)

        # Calculate L1 regularization on hidden layer activations, 
        # i.e. output of encoder, to encourage sparsity
        # in particular encoded is of shape (batch_size, hidden_size)
        
        l1_loss = torch.mean(torch.abs(encoded)) 
        # given the L1 norm of one sample this averages the L1 norm over
        # batch_size and hidden_size
        # Alternatively: can only average over batch_size and not hidden_size
        # what the code below does: for each sample in the batch,
        # calculate the L1 norm: sum of absolute values of the elements
        # then take mean of l1 norm over all samples
        #l1_loss = torch.mean(torch.sum(torch.abs(encoded), dim=1))
        # equivalently: l1_loss = torch.mean(torch.norm(encoded, p=1, dim=1))

        # number of non-zero elements, averaged over batch dimension
        #l0_loss = torch.mean((encoded != 0).sum(dim=1))

        # we make sure that decoded and targets have the expected shape, i.e., (batch_dimension, hidden_dimension)
        assert decoded.shape == targets.shape
        assert len(decoded.shape) == 2
        #assert decoded.shape[0] == self.batch_size --> we don't have access to the batch_size here so we do this in model_pipeline.py

        # we compute for each element the squared difference --> shape: (batch_size, vector_size)
        squared_differences = torch.square(decoded - targets)
        # for each vector dimension we compute the MSE over all samples in the batch --> shape: (vector_size)
        sample_MSE = torch.mean(squared_differences, dim=0)
        # we compute the difference of max and min for each dimension of the true data --> shape: (vector_size)
        # torch.max returns (max values, indices) --> we only want the values so we use [0]
        sample_range = torch.max(targets, dim=0)[0] - torch.min(targets, dim=0)[0]
        # we compute for each dimension, the NRMSE --> shape: (vector_size)
        sample_RMSE = torch.sqrt(sample_MSE) 
        sample_NRMSE = sample_RMSE / sample_range
        # We don't normalize the rmse by the mean of the targets, as the targets are not necessarily positive,
        # so the mean might be zero
        # in the case of conv activations, which are inputted 
        # we average the NRMSE over all dimensions --> shape: (1)
        nrmse = torch.mean(sample_NRMSE)
        rmse = torch.mean(sample_RMSE)

        return reconstruction_loss, l1_loss, nrmse, rmse