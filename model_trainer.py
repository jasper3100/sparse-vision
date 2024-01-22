import torch 

class ModelTrainer:
    def __init__(self,
                 model, 
                 optimizer, 
                 criterion):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0.0

        for inputs, targets in dataloader:
            #inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)

    def validate_epoch(self, dataloader):
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for inputs, targets in dataloader:
                #inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                total_loss += loss.item()

        return total_loss / len(dataloader)

    def train(self, train_dataloader, valid_dataloader=None, num_epochs):
        for epoch in range(num_epochs):
            train_loss = self.train_epoch(train_dataloader)
            
            if valid_dataloader is not None:
                valid_loss = self.validate_epoch(valid_dataloader)
                print(f'Epoch {epoch + 1}/{num_epochs} -> Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}')
            else:
                print(f'Epoch {epoch + 1}/{num_epochs} -> Train Loss: {train_loss:.4f}')

        print('Training complete.')

# Example usage:
# Assuming you have a PyTorch model, criterion, and optimizer already defined
# model = ...
# criterion = ...
# optimizer = ...
# trainer = SimpleTrainer(model, criterion, optimizer)
# trainer.train(train_dataloader, valid_dataloader, num_epochs)