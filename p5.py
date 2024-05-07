import torch
import torch.nn as nn
import torch.optim as optim
import psutil
import os
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms

# Function to get the current memory usage in MB
def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 2)  # in MB

# Define a simple autoencoder for demonstration
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Linear(256, 128)
        self.decoder = nn.Linear(128, 256)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Define a dummy main model
class MainModel(nn.Module):
    def __init__(self):
        super(MainModel, self).__init__()
        self.layer = nn.Linear(28*28, 256)  # Example layer

    def forward(self, x):
        return self.layer(x)

# Instantiate the models
main_model = MainModel()
autoencoder = Autoencoder()

# Optimizer for the autoencoder
ae_optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)

# Define a forward hook to train the autoencoder
def hook(module, inputs, outputs):
    global cumulative_rec_loss
    with torch.enable_grad():  # Enable gradients
        # Loss calculation for autoencoder
        # Assuming outputs from the main model are used as inputs to the autoencoder
        ae_outputs = autoencoder(outputs.detach())  # Detach to avoid gradients propagating back to the main model
        loss = nn.MSELoss()(ae_outputs, outputs)  # Reconstruction loss
        loss.backward()  # Backpropagate gradients to the autoencoder
        ae_optimizer.step()  # Update autoencoder
        ae_optimizer.zero_grad()  # Reset gradients for next step
        cumulative_rec_loss += loss.item()

# Register the forward hook on the main model's layer
main_model.layer.register_forward_hook(hook)

# Training loop for 5 epochs
num_epochs = 10
batch_size = 32  # Example batch size
train_shuffle = True
drop_last = True

root_dir='datasets/mnist'
directory_path = 'C:\\Users\\Jasper\\Downloads\\Master thesis\\Code'
root_dir=os.path.join(directory_path, root_dir)
download = not os.path.exists(root_dir)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_dataset = torchvision.datasets.MNIST(root_dir, train=True, download=download, transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=train_shuffle, drop_last=drop_last)

for epoch in range(num_epochs):
    # Monitor memory at the beginning of the epoch
    mem_before = get_memory_usage()
    print(f"Epoch {epoch + 1} - Memory Before: {mem_before:.2f} MB")

    cumulative_rec_loss = 0.0

    # Forward pass with torch.no_grad for the main model
    with torch.no_grad():
        for inputs, targets in train_dataloader:
            inputs = inputs.view(inputs.size(0), -1)  # Flatten the input to (batch_size, 28 * 28)
            outputs = main_model(inputs)

    print(f"Epoch {epoch + 1} - Cumulative Reconstruction Loss: {cumulative_rec_loss:.4f}")
    del inputs, outputs, cumulative_rec_loss
    
    # Monitor memory at the end of the epoch
    mem_after = get_memory_usage()
    print(f"Epoch {epoch + 1} - Memory After: {mem_after:.2f} MB")