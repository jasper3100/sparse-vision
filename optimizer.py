import torch
import torch.nn as nn

class Optimizer(nn.Module):
    def __init__(self, model, learning_rate, optimizer_name):
        super(Optimizer, self).__init__()
        self.optimizer_name = optimizer_name
        self.learning_rate = learning_rate
        self.model = model

    def forward(self):
        # Optimizer requires a forward function
        if self.optimizer_name == 'adam':
            return torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        elif self.optimizer_name == 'sgd':
            return torch.optim.SGD(self.model.parameters(), lr=self.learning_rate) # momentum= 0.9, weight_decay=1e-4)