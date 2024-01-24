import torch.nn as nn

from losses.sparse_loss import SparseLoss

class Criterion(nn.Module):
    def __init__(self, criterion_name):
        super(Criterion, self).__init__()
        self.criterion_name = criterion_name

    def forward(self, lambda_sparse=None):
        # Criterion requires a forward function
        if self.criterion_name == 'cross_entropy':
            return nn.CrossEntropyLoss()
        elif self.criterion_name == 'sae_loss':
            return SparseLoss(lambda_sparse=lambda_sparse)