import torch
import torch.nn as nn
import torch.optim as optim

# Create a simple dataset
# Inputs: [1, 1], [2, 2], [3, 3]
# Output: [2], [4], [6]
inputs = torch.tensor([[1, 1,1], [2, 2,2], [3, 3,3]], dtype=torch.float32)
outputs = torch.tensor([[2,2], [4,4], [6,6]], dtype=torch.float32)

# Define the model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(3,2)

    def forward(self, x):
        return self.linear(x)

# Initialize the model, loss function, and optimizer
model = SimpleModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
for epoch in range(1):
    # Forward pass
    predictions = model(inputs)

    # Calculate the loss
    loss = criterion(predictions, outputs)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print internal weights (moving averages) of Adam optimizer
    for group in optimizer.param_groups:
        for p in group['params']:
            if p.grad is not None:
                state = optimizer.state[p] # optimizer.state[bias[index]]
                print(state)
                if len(state['exp_avg'].shape)==2:
                    state['exp_avg'][0][0] = 0
                # for a linear layer
                # weight = optimizer.param_groups[0]['params'][0] is the weight matrix of the linear layer
                # bias = optimizer.param_groups[0]['params'][1] is the bias of the linear layer
            
                # with weight matrix of shape (out_features, in_features) and bias of shape (out_features)
                # the optimizer.state[weight]['exp_avg'] is of shape (out_features, in_features), i.e., it contains a moving average for each weight
                # optimizer.state[bias]['exp_avg'] is of shape (out_features), i.e., it contains a moving average for each bias
                # same holds for 'exp_avg_sq'
                else:
                    state['exp_avg'][0] = 1
                print(state)
                #print(f"Step {epoch + 1} - Mavg: {state['exp_avg']}, Mavg_sq: {state['exp_avg_sq']}")


