from train_process import TrainProcess
from model_saver import ModelSaver
from criterion import Criterion
from optimizer import Optimizer
from utils import load_model_aux, load_data_aux

class TrainingPipeline:
    '''
    A class that orchestrates and manages the training pipeline, including 
    loading models, data, setting up training parameters, and saving weights.
    '''
    def __init__(self, 
                 model_name,
                 dataset_name,
                 data_dir=None,
                 layer_name=None,
                 epochs=3,
                 learning_rate=0.001,
                 batch_size=32,
                 expansion_factor=None,
                 weights_folder_path=None):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.layer_name = layer_name
        self.expansion_factor = expansion_factor
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.weights_folder_path = weights_folder_path
        self.train_dataloader, self.valid_dataloader, self.img_size, _ = load_data_aux(dataset_name=dataset_name, 
                                                                                    batch_size=self.batch_size,
                                                                                    data_dir=data_dir, 
                                                                                    layer_name=layer_name)
        self.model, _ = load_model_aux(model_name, 
                                       self.img_size, 
                                       self.expansion_factor)

    def execute_training(self, 
              criterion_name,
              optimizer_name,
              lambda_sparse=None): 
        
        criterion_builder = Criterion(criterion_name)
        criterion = criterion_builder.forward(lambda_sparse)
        optimizer_builder = Optimizer(self.model, self.learning_rate, optimizer_name)
        optimizer = optimizer_builder.forward()

        train_process = TrainProcess(model=self.model, 
                                     optimizer = optimizer, 
                                     criterion = criterion)
        train_process.train(self.train_dataloader, self.epochs, self.valid_dataloader)

    def save_model_weights(self):
        model_saver = ModelSaver(self.model, self.weights_folder_path)
        model_saver.save_model_weights(layer_name=self.layer_name)


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