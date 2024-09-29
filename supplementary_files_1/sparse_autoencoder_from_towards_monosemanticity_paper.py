class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SparseAutoencoder, self).__init__()
        self.bias = nn.Parameter(torch.ones(1))
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=True),
            nn.ReLU()
        )
        self.decoder = nn.Linear(hidden_dim, input_dim, bias=True)

    def forward(self, x):
        x = x.view(x.size(0), -1) # flatten input: [10, 256, 56, 56] -> [10, 802816]
        print(x.size())
        x = x + self.bias
        print(self.bias)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        decoded = decoded.view(decoded.size(0), 256, 56, 56) # unflatten output: [10, 802816] -> [10, 256, 56, 56]
        print(decoded.size())
        return encoded, decoded

# define loss function
class SparseLoss(nn.Module):
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

if __name__ == "__main__":
    # Define input and hidden dimensions
    input_dim = int(256 * 56 * 56) # = 802816
    hidden_dim = int(input_dim * 2) 
    # In the "Towards Monosemanticity" paper, the largest hidden dimension they
    # use is ~131000. Our hidden_dim is much larger. In contrast to this paper
    # we consider vision models instead of language models.

    # Instantiate the sparse autoencoder (SAE) model
    sae = SparseAutoencoder(input_dim, hidden_dim)
    optimizer = torch.optim.Adam(sae.parameters(), lr=0.001)
    criterion = SparseLoss(lambda_sparse=0.1)

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        sae.train()
        optimizer.zero_grad()
        encoded, decoded = sae(activation)
        loss = criterion(encoded, decoded, activation)
        loss.backward()
        optimizer.step()
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")