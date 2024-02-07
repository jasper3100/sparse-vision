import torch 
import wandb

from utils import *
from evaluate_feature_maps import polysemanticity_level

class Training:
    '''
    Class that handles the training process.
    '''
    def __init__(self,
                 model, 
                 model_name,
                 device,
                 optimizer_name, 
                 criterion_name,
                 learning_rate,
                 lambda_sparse=None,
                 dataloader_2=None,
                 num_classes=None,
                 activation_threshold=None,
                 expansion_factor=None):
        self.model = model
        self.model_name = model_name
        self.device = device
        self.lambda_sparse = lambda_sparse
        self.dataloader_2 = dataloader_2
        self.num_classes = num_classes
        self.activation_threshold = activation_threshold
        self.expansion_factor = expansion_factor
        self.optimizer = get_optimizer(optimizer_name, self.model, learning_rate)
        self.criterion = get_criterion(criterion_name, lambda_sparse)

    def epoch(self, dataloader, is_train):
        if is_train:
            self.model.train()
        else: 
            self.model.eval()
        
        with torch.set_grad_enabled(is_train):
            total_loss = 0.0
            total_rec_loss = 0.0
            total_l1_loss = 0.0

            if self.dataloader_2 is None:
                for data in dataloader:
                    if isinstance(data, torch.Tensor) and 'sae' in self.model_name:
                        # if the dataloader doesn't contain targets, then we use
                        # the inputs as targets (f.e. autoencoder reconstruction loss)
                        inputs = data.to(self.device)
                        encoded, decoded = self.model(inputs)
                        rec_loss, l1_loss = self.criterion(encoded, decoded, inputs) # the inputs are the targets
                        loss = rec_loss + self.lambda_sparse*l1_loss
                        total_rec_loss += rec_loss.item()
                        total_l1_loss += l1_loss.item()
                    elif isinstance(data, (list, tuple)) and len(data) == 2 and 'sae' not in self.model_name:
                        inputs, targets = data
                        inputs, targets = inputs.to(self.device), targets.to(self.device)
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, targets)
                    else: 
                        raise ValueError("Unexpected combination of model and train data format")
                    if is_train:
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()
                    total_loss += loss.item()
                    
                return total_loss / len(dataloader), total_rec_loss / len(dataloader), total_l1_loss / len(dataloader)
            
            elif self.dataloader_2 is not None and 'sae' in self.model_name:
                total_mean_active_classes_per_neuron = 0.0
                total_std_active_classes_per_neuron = 0.0
                total_activated_units = 0.0
                total_total_units = 0.0
                for data, data_2 in zip(dataloader, self.dataloader_2):
                    # dataloader is expected to be the activations of the model (no targets provided, since the target is the input itself)
                    # dataloader_2 is expected to be the original train_dataloader, having inputs and targets --> we use the targets for computing the polysemanticity level
                    if isinstance(data, torch.Tensor) and isinstance(data_2, (list, tuple)) and len(data_2) == 2:
                        inputs = data.to(self.device)
                        encoded, decoded = self.model(inputs)
                        rec_loss, l1_loss = self.criterion(encoded, decoded, inputs) # the inputs are the targets
                        loss = rec_loss + self.lambda_sparse*l1_loss

                        inputs, targets = data_2
                        inputs, targets = inputs.to(self.device), targets.to(self.device)
                        mean_active_classes_per_neuron, std_active_classes_per_neuron = polysemanticity_level(encoded, targets, self.num_classes, self.activation_threshold)

                        activated_units, total_units = measure_activating_units(encoded, self.activation_threshold) 
                    else:
                        raise ValueError("Unexpected combination of model and train data format")
                    if is_train:
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()
                    total_loss += loss.item()
                    total_rec_loss += rec_loss.item()
                    total_l1_loss += l1_loss.item()
    
                    total_mean_active_classes_per_neuron += mean_active_classes_per_neuron
                    total_std_active_classes_per_neuron += std_active_classes_per_neuron
                    total_activated_units += activated_units
                    total_total_units += total_units

                return total_loss / len(dataloader), total_mean_active_classes_per_neuron / len(dataloader), total_std_active_classes_per_neuron / len(dataloader), total_activated_units, total_total_units, total_rec_loss / len(dataloader), total_l1_loss / len(dataloader)

    def train(self, 
              train_dataloader, 
              num_epochs, 
              wandb_status, 
              valid_dataloader=None, 
              train_dataset_length=None, 
              valid_dataset_length=None, 
              folder_path=None,
              layer_names=None,
              params=None):
        print('Training started.')
        for epoch in range(num_epochs):
            if self.dataloader_2 is None and valid_dataloader is None:
                train_loss, train_rec_loss, train_l1_loss = self.epoch(train_dataloader, is_train=True)
                if wandb_status:
                    wandb.log({f"{self.model_name}_train_loss": train_loss})
                print(f'Epoch {epoch + 1}/{num_epochs} -> Train Loss: {train_loss:.4f}')
            elif self.dataloader_2 is None and valid_dataloader is not None:
                train_loss, train_rec_loss, train_l1_loss = self.epoch(train_dataloader, is_train=True)
                valid_loss, val_rec_loss, val_l1_loss = self.epoch(valid_dataloader, is_train=False)
                if wandb_status:
                    wandb.log({f"{self.model_name}_train_loss": train_loss, f"{self.model_name}_val_loss": valid_loss})
                print(f'Epoch {epoch + 1}/{num_epochs} -> Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}')
            elif self.dataloader_2 is not None and valid_dataloader is None and 'sae' in self.model_name:
                train_loss, mean_active_classes_per_neuron, std_active_classes_per_neuron, activated_units, total_units, train_rec_loss, train_l1_loss = self.epoch(train_dataloader, is_train=True)
                train_sae_sparsity, train_sae_mean_activated_units, train_sae_mean_total_units = compute_sparsity(activated_units, total_units, train_dataset_length)
                if wandb_status:
                    wandb.log({f"{self.model_name}_train_loss": train_loss,
                               "train_mean_active_classes_per_neuron": mean_active_classes_per_neuron, 
                               "train_mean+std_active_classes_per_neuron": mean_active_classes_per_neuron + std_active_classes_per_neuron,
                               "train_mean-std_active_classes_per_neuron": mean_active_classes_per_neuron - std_active_classes_per_neuron,
                               "train_sae_sparsity": train_sae_sparsity,
                               "train_sae_mean_activated_units": train_sae_mean_activated_units,
                               "train_sae_mean_total_units": train_sae_mean_total_units})
                print(f'Epoch {epoch + 1}/{num_epochs} -> Train loss: {train_loss:.4f},', 
                      f'Train Mean active classes per neuron: {mean_active_classes_per_neuron:.4f},', 
                      f'Train Std active classes per neuron: {std_active_classes_per_neuron:.4f},',
                      f'Train SAE sparsity: {train_sae_sparsity:.4f},',
                      f'Train SAE mean activated units: {train_sae_mean_activated_units},',
                      f'Train SAE mean total units: {train_sae_mean_total_units}')
            elif self.dataloader_2 is not None and valid_dataloader is not None and 'sae' in self.model_name:
                train_loss, train_mean_active_classes_per_neuron, train_std_active_classes_per_neuron, train_activated_units, train_total_units, train_rec_loss, train_l1_loss = self.epoch(train_dataloader, is_train=True)
                valid_loss, val_mean_active_classes_per_neuron, val_std_active_classes_per_neuron, val_activated_units, val_total_units, val_rec_loss, val_l1_loss = self.epoch(valid_dataloader, is_train=False)
                train_sae_sparsity, train_sae_mean_activated_units, train_sae_mean_total_units = compute_sparsity(train_activated_units, train_total_units, train_dataset_length)
                val_sae_sparsity, val_sae_mean_activated_units, val_sae_mean_total_units = compute_sparsity(val_activated_units, val_total_units, valid_dataset_length)
                if wandb_status:
                    wandb.log({f"{self.model_name}_train_loss": train_loss, 
                               f"{self.model_name}_val_loss": valid_loss,
                               "train_mean_active_classes_per_neuron": train_mean_active_classes_per_neuron, 
                               "train_mean+std": train_mean_active_classes_per_neuron + train_std_active_classes_per_neuron,
                               "train_mean-std": train_mean_active_classes_per_neuron - train_std_active_classes_per_neuron,
                               "val_mean_active_classes_per_neuron": val_mean_active_classes_per_neuron,
                               "val_mean+std_active_classes_per_neuron": val_mean_active_classes_per_neuron + val_std_active_classes_per_neuron,
                               "val_mean-std_active_classes_per_neuron": val_mean_active_classes_per_neuron - val_std_active_classes_per_neuron,
                               "train_sae_sparsity": train_sae_sparsity,
                               "train_sae_mean_activated_units": train_sae_mean_activated_units,
                               "train_sae_mean_total_units": train_sae_mean_total_units,
                               "val_sae_sparsity": val_sae_sparsity,
                               "val_sae_mean_activated_units": val_sae_mean_activated_units,
                               "val_sae_mean_total_units": val_sae_mean_total_units})
                print(f'Epoch {epoch + 1}/{num_epochs} ->',
                      f'Train loss: {train_loss:.4f},', 
                      f'Valid loss: {valid_loss:.4f},',
                        f'Train Mean active classes per neuron: {train_mean_active_classes_per_neuron:.4f},',
                        f'Train Std active classes per neuron: {train_std_active_classes_per_neuron:.4f},',
                        f'Valid Mean active classes per neuron: {val_mean_active_classes_per_neuron:.4f},',
                        f'Valid Std active classes per neuron: {val_std_active_classes_per_neuron:.4f},',
                        f'Train SAE sparsity: {train_sae_sparsity:.4f},',
                        f'Train SAE mean activated units: {train_sae_mean_activated_units},',
                        f'Train SAE mean total units: {train_sae_mean_total_units},',
                        f'Valid SAE sparsity: {val_sae_sparsity:.4f},',
                        f'Valid SAE mean activated units: {val_sae_mean_activated_units},',
                        f'Valid SAE mean total units: {val_sae_mean_total_units}') 
            else:
                raise ValueError("Unexpected combination of dataloaders and models")
            
        print('Training complete.')

        if 'sae' in self.model_name:
            # We store the train_rec_loss and train_l1_loss from the last epoch
            store_sae_losses(folder_path, layer_names, params, self.lambda_sparse, self.expansion_factor, train_rec_loss, train_l1_loss)

    def save_model(self, weights_folder_path, layer_names=None, params=None):
        # layer_name is used for SAE models, because SAE is trained on activations
        # of a specific layer
        save_model_weights(self.model, weights_folder_path, layer_names=layer_names, params=params)