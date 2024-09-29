# EXAMPLE CODE ON HOW TO CHECKPOINT A MODEL WHERE SAE IS TRAINED IN FORWARD HOOK
# Eval loss before and after checkpointing should be the same!!!

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init

# Base model
class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.linear1 = nn.Linear(10, 10)
        self.linear2 = nn.Linear(10, 1)	
                                          
    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x

# Autoencoder
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.linear = nn.Linear(10, 1)
        self.linear.weight = nn.Parameter(init.kaiming_uniform_(torch.empty(1, 10)))
        self.linear.bias = nn.Parameter(torch.zeros(1))
        self.decoder = nn.Linear(1, 10)
                                          
    def forward(self, x):
        x = self.linear(x)
        x = self.decoder(x)
        return x

def hook_fct(output):
    print(train)
    if train:
        optimizer.zero_grad()
        ae_output = model(output)
        loss = nn.functional.mse_loss(output, ae_output)
        loss.backward()
        optimizer.step()
    if not train: 
        with torch.no_grad():
            ae_output = model(output)
    output = ae_output

# Create toy data
inputs = torch.randn(100, 10)
targets = torch.randn(100, 1)
inputs_part1 = inputs[:50]
targets_part1 = targets[:50]
inputs_part2 = inputs[50:]
targets_part2 = targets[50:]

# Initialize model and optimizer
model = MyModel()
model.train()
for param in model.parameters():
    param.requires_grad = True
optimizer = optim.Adam(model.parameters(), lr=0.001)

base_model = BaseModel()
base_model.eval()
for param in base_model.parameters():
    param.requires_grad = False
base_model.linear1.register_forward_hook(lambda module, input, output: hook_fct(output))

# Train for one epoch
for epoch in range(1):
    train = True
    _ = base_model(inputs_part1)

# Eval for one epoch
with torch.no_grad():
    train = False
    outputs = base_model(inputs)
    loss = nn.functional.mse_loss(outputs, targets)
    print(f'eval loss before checkpoint: {loss.item()}')

# Save checkpoint
torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'base_model_state_dict': base_model.state_dict(),
            #'loss': loss,
            }, 'checkpoint.pth')

# Train for one more epoch
for epoch in range(1):
    train = True
    _ = base_model(inputs_part2)
    
model = None 
base_model = None
optimizer = None

# Load checkpoint and continue training for one more epoch
checkpoint = torch.load('checkpoint.pth')
model = MyModel()
model.load_state_dict(checkpoint['model_state_dict'])
base_model = BaseModel()
base_model.load_state_dict(checkpoint['base_model_state_dict'])
optimizer = optim.Adam(model.parameters(), lr=0.001)
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
model.train()
for param in model.parameters():
    param.requires_grad = True
base_model.eval()
for param in base_model.parameters():
    param.requires_grad = False
base_model.linear1.register_forward_hook(lambda module, input, output: hook_fct(output))

with torch.no_grad():
    train = False
    outputs = base_model(inputs)
    loss = nn.functional.mse_loss(outputs, targets)
    print(f'eval loss after checkpoint: {loss.item()}')

for epoch in range(checkpoint['epoch'] + 1, checkpoint['epoch'] + 2):
    train = True
    _ = base_model(inputs_part2)