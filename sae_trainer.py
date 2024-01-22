import torch
import os

from sae import SparseAutoencoder
from sparse_loss import SparseLoss
from model_trainer import ModelTrainer
from model_saver import ModelSaver
from utils import load_feature_map

class SAETrainer:
    def __init__(self, 
                 original_activations_folder_path, 
                 layer_name, 
                 sae_weights_folder_path, 
                 expansion_factor, 
                 lambda_sparse=0.1,
                 epochs=3,
                 learning_rate=0.001):
        self.original_activations_folder_path = original_activations_folder_path
        self.layer_name = layer_name
        self.weights_folder_path = sae_weights_folder_path
        self.expansion_factor = expansion_factor
        self.lambda_sparse = lambda_sparse
        self.epochs = epochs
        self.learning_rate = learning_rate

    def train_sae(self): 
        file_path = os.path.join(self.original_activations_folder_path, f'{self.layer_name}_activations.h5')
        train_dataloader = load_feature_map(file_path).float()



        criterion = SparseLoss(lambda_sparse=self.lambda_sparse) 
        model = SparseAutoencoder(img_size, self.expansion_factor)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)

        trainer = ModelTrainer(model, criterion, optimizer)
        trainer.train(train_dataloader, valid_dataloader, num_epochs)

        #encoded, decoded = sae(data)
        #loss = criterion(encoded, decoded, data)

        model_saver = ModelSaver(model, self.weights_folder_path)
        model_saver.save_model_weights()

'''
if __name__ == "__main__":
    activations_folder_path = # specify your path
    layer_name = 'model.layer1[0].conv3'
    sae_weights_folder_path = # specify your path
    expansion_factor = 2
    lambda_sparse = 0.1

    sae_trainer = SAETrainer(activations_folder_path, layer_name, sae_weights_folder_path, expansion_factor, lambda_sparse)
    sae_trainer.train_sae()
'''