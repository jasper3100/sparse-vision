import torch 
import time
import wandb

from utils import get_criterion, save_model_weights, get_optimizer

class Training:
    '''
    Class that handles the training process.
    '''
    def __init__(self,
                 model, 
                 device,
                 optimizer_name, 
                 criterion_name,
                 learning_rate,
                 lambda_sparse=None):
        self.model = model
        self.device = device
        self.optimizer = get_optimizer(optimizer_name, self.model, learning_rate)
        self.criterion = get_criterion(criterion_name, lambda_sparse)

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0.0
        idx = 0
        for data in dataloader:
            #print(data.shape)
            if isinstance(data, (list, tuple)) and len(data) == 2:
                inputs, targets = data
                #print(inputs.shape, targets.shape)
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                idx += 1
                if idx == 10:
                    break
            elif isinstance(data, torch.Tensor):
                # if the dataloader doesn't contain targets, then we use
                # the inputs as targets (f.e. autoencoder reconstruction loss)
                inputs = data.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, inputs)
                idx += 1
                if idx == 10:
                    break
            else:
                raise ValueError("Unexpected data format from dataloader")
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            print("One batch done.")

        return total_loss / len(dataloader)

    def validate_epoch(self, dataloader):
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for data in dataloader:
                #data = data.to(self.device)
                if isinstance(data, (list, tuple)) and len(data) == 2:
                    inputs, targets = data
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                elif isinstance(data, torch.Tensor):
                    # if the dataloader doesn't contain targets, then we use
                    # the inputs as targets (f.e. autoencoder reconstruction loss)
                    inputs = data.to(self.device)
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, inputs)
                else:
                    raise ValueError("Unexpected data format from dataloader")
                
                total_loss += loss.item()

        return total_loss / len(dataloader)

    def train(self, train_dataloader, num_epochs, name, valid_dataloader=None):
        print('Training started.')
        for epoch in range(num_epochs):
            train_loss = self.train_epoch(train_dataloader)
            
            if valid_dataloader is not None:
                valid_loss = self.validate_epoch(valid_dataloader)
                if name == "model":
                    wandb.log({"model_train_loss": train_loss, "model_val_loss": valid_loss})
                elif name == "sae":
                    wandb.log({"sae_train_loss": train_loss, "sae_val_loss": valid_loss})
                else:
                    raise ValueError(f"Unexpected name: {name}")
                print(f'Epoch {epoch + 1}/{num_epochs} -> Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}')
            else:
                if name == "model":
                    wandb.log({"model_train_loss": train_loss})
                elif name == "sae":
                    wandb.log({"sae_train_loss": train_loss})
                else:
                    raise ValueError(f"Unexpected name: {name}")
                print(f'Epoch {epoch + 1}/{num_epochs} -> Train Loss: {train_loss:.4f}')

        print('Training complete.')

    def save_model(self, weights_folder_path, layer_name=None, params=None):
        # layer_name is used for SAE models, because SAE is trained on activations
        # of a specific layer
        save_model_weights(self.model, weights_folder_path, layer_name=layer_name, params=params)