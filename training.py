import torch 

from utils import get_criterion, save_model_weights, load_model
from optimizer import Optimizer

class Training:
    '''
    Class that handles the training process.
    '''
    def __init__(self,
                 model, 
                 optimizer_name, 
                 criterion_name,
                 learning_rate,
                 lambda_sparse=None):
        self.model = model
        optimizer_builder = Optimizer(self.model, learning_rate, optimizer_name)
        self.optimizer = optimizer_builder.forward()
        self.criterion = get_criterion(criterion_name, lambda_sparse)

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0.0

        for data in dataloader:
            if isinstance(data, (list, tuple)) and len(data) == 2:
                inputs, targets = data
                #inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
            elif isinstance(data, torch.Tensor):
                # if the dataloader doesn't contain targets, then we use
                # the inputs as targets (f.e. autoencoder reconstruction loss)
                inputs = data
                outputs = self.model(inputs)
                loss = self.criterion(outputs, inputs)
            else:
                raise ValueError("Unexpected data format from dataloader")
            
            self.optimizer.zero_grad()
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

    def train(self, train_dataloader, num_epochs, valid_dataloader=None):
        print('Training started.')
        for epoch in range(num_epochs):
            train_loss = self.train_epoch(train_dataloader)
            
            if valid_dataloader is not None:
                valid_loss = self.validate_epoch(valid_dataloader)
                print(f'Epoch {epoch + 1}/{num_epochs} -> Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}')
            else:
                print(f'Epoch {epoch + 1}/{num_epochs} -> Train Loss: {train_loss:.4f}')

        print('Training complete.')

    def save_model(self, weights_folder_path, layer_name=None):
        # layer_name is used for SAE models, because SAE is trained on activations
        # of a specific layer
        save_model_weights(self.model, weights_folder_path, layer_name=layer_name)