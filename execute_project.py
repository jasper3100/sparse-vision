import ast

from utils import *
from model_pipeline import ModelPipeline
from evaluation import Evaluation

class ExecuteProject:
    def __init__(self,
                 model_name=None,
                sae_model_name=None,
                layer_names=None,
                directory_path=None,
                wandb_status=None,
                model_epochs=None,
                model_learning_rate=None,
                batch_size=None,
                model_optimizer_name=None,
                sae_epochs=None,
                sae_learning_rate=None,
                sae_optimizer_name=None,
                sae_batch_size=None,
                sae_lambda_sparse=None,
                sae_expansion_factor=None,
                activation_threshold=None,
                dataset_name=None,
                use_sae=None,
                train_sae=None,
                train_original_model=None,
                store_activations=None,
                compute_feature_similarity=None,
                model_criterion_name=None,
                sae_criterion_name=None,
                run_group_ID=None,
                dead_neurons_epochs=None):
        self.model_name = model_name
        self.sae_model_name = sae_model_name
        self.layer_names = ast.literal_eval(layer_names) # turn the string ['fc1'] into an actual list
        self.directory_path = directory_path
        self.wandb_status = eval(wandb_status) # Turn 'False' into False, 'True' into True
        # If we run the code locally and use the parameters specified in the txt file, we need to convert
        # some of them from string into the desired format
        self.model_epochs = int(model_epochs) if model_epochs is not None else None
        self.model_learning_rate = float(model_learning_rate) if model_learning_rate is not None else None
        self.batch_size = int(batch_size) if batch_size is not None else None
        self.model_optimizer_name = model_optimizer_name
        self.sae_epochs = int(sae_epochs) if sae_epochs is not None else None
        self.sae_learning_rate = float(sae_learning_rate) if sae_learning_rate is not None else None
        self.sae_optimizer_name = sae_optimizer_name
        self.sae_batch_size = int(sae_batch_size) if sae_batch_size is not None else None
        self.sae_lambda_sparse = float(sae_lambda_sparse) if sae_lambda_sparse is not None else None
        self.sae_expansion_factor = float(sae_expansion_factor) if sae_expansion_factor is not None else None
        self.activation_threshold = float(activation_threshold) if activation_threshold is not None else None
        self.dataset_name = dataset_name
        self.use_sae = eval(use_sae) if use_sae is not None else None
        self.train_sae = eval(train_sae) if train_sae is not None else None
        self.train_original_model = eval(train_original_model) if train_original_model is not None else None
        self.store_activations = eval(store_activations) if store_activations is not None else None
        self.compute_feature_similarity = eval(compute_feature_similarity) if compute_feature_similarity is not None else None
        self.model_criterion_name = model_criterion_name
        self.sae_criterion_name = sae_criterion_name
        self.run_group_ID = run_group_ID
        self.dead_neurons_epochs = dead_neurons_epochs

        # These parameter dictionaries are used for creating file names, f.e., to store model weights, feature maps, etc. Hence, include any parameter here that you would like to 
        # be included in the file names to better use and identify files, model_name and dataset_name are already considered
        self.model_params = {'epochs': model_epochs, 'learning_rate': model_learning_rate, 'batch_size': batch_size, 'optimizer': model_optimizer_name, 'activation_threshold': activation_threshold}
        self.sae_params = {'epochs': sae_epochs, 'learning_rate': sae_learning_rate, 'batch_size': sae_batch_size, 'optimizer': sae_optimizer_name, 'expansion_factor': sae_expansion_factor, 
                           'lambda_sparse': sae_lambda_sparse, 'activation_threshold': activation_threshold, 'dead_neurons_epochs': dead_neurons_epochs}
        # sae_params but without lambda_sparse and expansion_factor --> is used to create files collecting results varying over these two parameters
        self.sae_params_1 = self.sae_params.copy()
        self.sae_params_1.pop('lambda_sparse', None)
        self.sae_params_1.pop('expansion_factor', None)

        self.model_weights_folder_path, self.sae_weights_folder_path, self.activations_folder_path, self.evaluation_results_folder_path = get_folder_paths(self.directory_path, self.model_name, self.dataset_name, self.sae_model_name)

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            print('Using GPU')
        else:
            self.device = torch.device('cpu')
            print('Using CPU')

        if self.train_sae:
            id = "train_sae"
        elif self.use_sae:
            id = "modified_model"
        elif self.train_original_model:
            id = "original_model"
        else:
            id = "train_original_model"

        self.run_group_ID = id + "_" + self.dataset_name + "_" + self.run_group_ID
        self.run_ID = get_file_path(layer_names=self.layer_names, params=self.model_params, params2=self.sae_params, file_name=self.run_group_ID)

        # load data loader
        self.train_dataloader, self.val_dataloader, self.category_names, self.img_size = load_data(self.directory_path, self.dataset_name, self.batch_size)
        # num_batches can be set to a different value if we want to limit the number of batches (which can be used wherever desired)

        if self.wandb_status:
            print("Logging to W&B")
            wandb.login()
            wandb.init(project="master-thesis",
                        name=self.run_ID,
                        group=self.run_group_ID,
                        #job_type="train", can specify job type for adding description
                        config={"run_ID": self.run_ID,
                                "run_group_ID": self.run_group_ID,
                                "model_name": self.model_name,
                                "sae_model_name": self.sae_model_name,
                                "dataset_name": self.dataset_name,
                                "layer_names": self.layer_names,
                                "sae_expansion_factor": self.sae_expansion_factor,
                                "directory_path": self.directory_path,
                                "model_epochs": self.model_epochs,
                                "model_learning_rate": self.model_learning_rate,
                                "model_optimizer_name": self.model_optimizer_name,
                                "sae_epochs": self.sae_epochs,
                                "sae_learning_rate": self.sae_learning_rate,
                                "sae_optimizer": self.sae_optimizer_name,
                                "batch_size": self.batch_size,
                                "sae_batch_size": self.sae_batch_size,
                                "sae_lambda_sparse": self.sae_lambda_sparse,
                                "activation_threshold": self.activation_threshold,
                                "use_sae": self.use_sae,
                                "train_sae": self.train_sae,
                                "train_original_model": self.train_original_model,
                                "store_activations": self.store_activations,
                                "compute_feature_similarity": self.compute_feature_similarity,
                                "model_criterion_name": self.model_criterion_name,
                                "sae_criterion_name": self.sae_criterion_name,
                                "dead_neurons_epochs": self.dead_neurons_epochs})

    def model_pipeline(self):
        pipeline = ModelPipeline(device=self.device,
                                train_dataloader=self.train_dataloader,
                                category_names=self.category_names,
                                layer_names=self.layer_names, 
                                activation_threshold=self.activation_threshold,
                                prof=None,
                                use_sae=self.use_sae,
                                train_sae=self.train_sae,
                                train_original_model=self.train_original_model,
                                store_activations=self.store_activations,
                                compute_feature_similarity=self.compute_feature_similarity,
                                activations_folder_path=self.activations_folder_path,
                                sae_weights_folder_path=self.sae_weights_folder_path,
                                model_weights_folder_path=self.model_weights_folder_path,
                                evaluation_results_folder_path=self.evaluation_results_folder_path)
        pipeline.instantiate_models(model_name=self.model_name, 
                                    img_size=self.img_size, 
                                    model_optimizer_name=self.model_optimizer_name,
                                    model_criterion_name=self.model_criterion_name,
                                    model_learning_rate=self.model_learning_rate,
                                    model_params=self.model_params,
                                    sae_model_name=self.sae_model_name,
                                    sae_expansion_factor=self.sae_expansion_factor,
                                    sae_lambda_sparse=self.sae_lambda_sparse,
                                    sae_optimizer_name=self.sae_optimizer_name,
                                    sae_criterion_name=self.sae_criterion_name,
                                    sae_learning_rate=self.sae_learning_rate,
                                    sae_params=self.sae_params,
                                    sae_params_1=self.sae_params_1)
        pipeline.deploy_model(num_epochs=self.model_epochs, 
                              dead_neurons_epochs=self.dead_neurons_epochs,
                            wandb_status=self.wandb_status)
                            
    def evaluation(self):
        evaluation = Evaluation(layer_names=self.layer_names,
                                wandb_status=self.wandb_status,
                                sae_params_1=self.sae_params_1,
                                evaluation_results_folder_path=self.evaluation_results_folder_path)
        evaluation.get_sae_eval_results()