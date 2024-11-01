import ast

from utils import *
from model_pipeline import ModelPipeline
from evaluation import Evaluation

class ExecuteProject:
    def __init__(self,
                 model_name=None,
                sae_model_name=None,
                sae_layer=None,
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
                dataset_name=None,
                original_model=None,
                training=None,
                model_criterion_name=None,
                sae_criterion_name=None,
                run_group_ID=None,
                dead_neurons_steps=None,
                run_evaluation=None,
                execution_location=None,
                mis=None,
                compute_ie=None,
                sae_checkpoint_epoch=None):
        self.model_name = model_name
        self.sae_model_name = sae_model_name
        self.sae_layer = sae_layer
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
        self.dataset_name = dataset_name
        self.use_sae = not eval(original_model) if original_model is not None else None
        self.training = eval(training) if training is not None else None
        self.model_criterion_name = model_criterion_name
        self.sae_criterion_name = sae_criterion_name
        self.run_group_ID = run_group_ID
        self.dead_neurons_steps = int(dead_neurons_steps)
        self.run_evaluation = run_evaluation if run_evaluation is not None else None
        self.execution_location = execution_location
        self.sae_checkpoint_epoch = int(sae_checkpoint_epoch) if sae_checkpoint_epoch is not None else None
        self.mis = mis # string
        self.compute_ie = compute_ie # string
        # WHEN ADDING A NEW PARAMETER HERE DONT FORGET TO TURN IT FROM STRING INTO THE DESIRED DATA FORMAT EVALUATE IT, i.e., int(x), float(x), eval(x), etc.

        if self.sae_checkpoint_epoch is not None:
            if self.sae_checkpoint_epoch > self.sae_epochs:
                raise ValueError("The SAE checkpoint epoch is greater than the number of SAE epochs.")
            if self.sae_checkpoint_epoch > 0:
                resume = "must"
                # for the different behaviors of "resume", see: https://docs.wandb.ai/ref/python/init 
            else:
                resume = None
        else:
            resume = None

        # These parameter dictionaries are used for creating file names, f.e., to store model weights, feature maps, etc. Hence, include any parameter here that you would like to 
        # be included in the file names to better use and identify files, model_name and dataset_name are already considered
        self.model_params = {'model_name': model_name, 'epochs': model_epochs, 'learning_rate': model_learning_rate, 'batch_size': batch_size, 'optimizer': model_optimizer_name}
        self.sae_params = {'sae_model_name': sae_model_name, 'sae_epochs': sae_epochs, 'learning_rate': sae_learning_rate, 'batch_size': sae_batch_size, 'optimizer': sae_optimizer_name, 'expansion_factor': sae_expansion_factor, 
                           'lambda_sparse': sae_lambda_sparse, 'dead_neurons_steps': dead_neurons_steps}
        # sae_params_1 is used to create files collecting results varying over the below parameters
        self.sae_params_1 = self.sae_params.copy()
        self.sae_params_1.pop('lambda_sparse', None)
        self.sae_params_1.pop('expansion_factor', None)
        self.sae_params_1.pop('batch_size', None)
        self.sae_params_1.pop('optimizer', None)
        self.sae_params_1.pop('learning_rate', None)
        self.sae_params_1.pop('sae_epochs', None)
        # sae_params_2 is used for run ID
        self.sae_params_2 = self.sae_params.copy()
        self.sae_params_2.pop('sae_epochs', None)

        print("-------------------")
        print(self.model_params)
        print(self.sae_params)
        print("-------------------")

        self.model_weights_folder_path, self.sae_weights_folder_path, self.evaluation_results_folder_path = get_folder_paths(self.directory_path, self.model_name, self.dataset_name, self.sae_model_name)

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            print('Using GPU')
        else:
            self.device = torch.device('cpu')
            print('Using CPU')

        
        if not self.use_sae and self.training:
            id = "train_original_model"
            if self.model_name == 'resnet18':
                raise ValueError("The module name 'resnet18' refers to the frozen and pre-trained model. Training it is not possible.")
        elif not self.use_sae and not self.training:
            id = "original_model"
            if self.model_name == 'resnet18_1' or self.model_name == 'resnet18_2':
                raise ValueError("The module names 'resnet18_1' or 'resnet18_2' refers to the trainable ResNet models. For inference please use the module name 'resnet18' to access the frozen model.") 
        elif self.use_sae and self.training:            
            id = f"train_sae_{self.sae_layer}"
            #if self.sae_optimizer_name != "constrained_adam":
            #    raise ValueError("When training SAEs, the optimizer must be constrained Adam to ensure decoder weights have unit norm.")
        elif self.use_sae and not self.training and not self.run_evaluation:
            id = f"modified_model_{self.sae_layer}"
        elif self.use_sae and not self.training and self.run_evaluation:
            id = "sae_evaluation"
        else:
            raise ValueError("The combination of parameters is not supported.")
               

        self.run_group_ID = id + "_" + self.dataset_name + "_" + self.run_group_ID
        if self.use_sae:
            # we don't include the number of epochs in the run ID because when using a checkpoint we might use a different number of epochs but we want to have the same run ID
            # in order to continue the run
            self.run_ID = get_file_path(sae_layer=self.sae_layer, params=self.model_params, params2=self.sae_params_2) #file_name=self.run_group_ID)
            # we use the appropriate number of epochs in pipeline.deploy_model()
            self.num_epochs = self.sae_epochs
        else:
            # if we only use the original model, the sae parameters are not included in the run ID & file names
            self.run_ID = get_file_path(sae_layer=self.sae_layer, params=self.model_params) #, file_name=self.run_group_ID)
            self.num_epochs = self.model_epochs

        print("Run ID: ", self.run_ID)

        if self.wandb_status and mis != "1" and compute_ie != "1": # if mis=1 or compute_ie=1, we just store some values to disk
            print("Logging to W&B")
            wandb.login()
            wandb.init(project="master-thesis",
                        name=self.run_ID, # for displaying name of run in W&B
                        id=self.run_ID, # used for resuming runs
                        resume=resume,
                        group=self.run_group_ID,
                        #job_type="train", can specify job type for adding description
                        config={"run_ID": self.run_ID,
                                "run_group_ID": self.run_group_ID,
                                "model_name": self.model_name,
                                "sae_model_name": self.sae_model_name,
                                "dataset_name": self.dataset_name,
                                "sae_layer": self.sae_layer,
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
                                "original_model": not self.use_sae,
                                "training": self.training,
                                "run_evaluation": self.run_evaluation,
                                "model_criterion_name": self.model_criterion_name,
                                "sae_criterion_name": self.sae_criterion_name,
                                "dead_neurons_steps": self.dead_neurons_steps,
                                "mis": self.mis,
                                "compute_ie": self.compute_ie,
                                "sae_checkpoint_epoch": self.sae_checkpoint_epoch})
            # we proceeed according to: https://docs.wandb.ai/guides/technical-faq/metrics-and-performance
            wandb.define_metric("batch")
            wandb.define_metric("epoch")
            # set all other train/ metrics to use this step (see here: https://docs.wandb.ai/guides/track/log/customize-logging-axes )
            wandb.define_metric("train/*", step_metric="batch")
            wandb.define_metric("eval/*", step_metric="epoch")
            # Alternatively, in the W&B user interface, I can choose the appropriate x-axis for a certain metric, or I can

    def model_pipeline(self):
        pipeline = ModelPipeline(device=self.device,
                                sae_layer=self.sae_layer, 
                                wandb_status=self.wandb_status,
                                prof=None,
                                use_sae=self.use_sae,
                                training=self.training,
                                sae_weights_folder_path=self.sae_weights_folder_path,
                                model_weights_folder_path=self.model_weights_folder_path,
                                evaluation_results_folder_path=self.evaluation_results_folder_path,
                                dead_neurons_steps=self.dead_neurons_steps,
                                sae_batch_size=self.sae_batch_size,
                                batch_size=self.batch_size,
                                dataset_name=self.dataset_name,
                                directory_path=self.directory_path,
                                mis=self.mis,
                                compute_ie=self.compute_ie)
        pipeline.instantiate_models(model_name=self.model_name, 
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
                                    sae_params_1=self.sae_params_1,
                                    execution_location=self.execution_location,
                                    sae_checkpoint_epoch=self.sae_checkpoint_epoch)
        pipeline.deploy_model(num_epochs=self.num_epochs)
                            
    def evaluation(self):
        evaluation = Evaluation(sae_layer=self.sae_layer,
                                wandb_status=self.wandb_status,
                                model_params=self.model_params,
                                sae_params_1=self.sae_params_1,
                                evaluation_results_folder_path=self.evaluation_results_folder_path,
                                sae_checkpoint_epoch=self.sae_checkpoint_epoch)
        #evaluation.plot_rec_loss_vs_sparsity(type_of_rec_loss="mse")
        #evaluation.plot_rec_loss_vs_sparsity(type_of_rec_loss="rmse")
        # evaluation.plot_rec_loss_vs_sparsity(type_of_rec_loss="nrmse")
        evaluation.plot_rec_loss_vs_sparsity_all_epochs(type_of_rec_loss="nrmse")
        #evaluation.compute_sae_ranking()

        if self.wandb_status:
            wandb.log({}, commit=True) # commit the logs from before