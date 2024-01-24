import os
import argparse

from activations_handler import ActivationsHandler
from train_pipeline import TrainingPipeline
from evaluate_model import ModelEvaluator

# I can run main.py as in the line below (or if I leave the arguments empty, it will use the default values)
# python main.py --store_activations --train_sae --modify_and_store_activations --model_name resnet18 --dataset_name cifar10 --layer_name model.layer2[0].conv1 --expansion_factor 3 --directory_path C:\Users\Jasper\Downloads\Master thesis\Code
# python main.py --model_name resnet50 --dataset_name tiny_imagenet --layer_name model.layer1[0].conv3 --expansion_factor 2 --directory_path C:\Users\Jasper\Downloads\Master thesis\Code --metrics ce percentage_same_classification

# IS THIS A GOOD MAIN.PY FILE?? OR SHOULD I ALSO USE CLASSES HERE??
# IS THIS THE RIGHT WAY TO USE PARAMETERS???
def parse_arguments():
    parser = argparse.ArgumentParser(description="Setting parameters")

    # command-line arguments
    parser.add_argument('--model_name', type=str, default='resnet50', help='Specify the model name')
    parser.add_argument('--sae_model_name', type=str, default='sae_mlp', help='Specify the sae model name')
    parser.add_argument('--dataset_name', type=str, default='tiny_imagenet', help='Specify the dataset name')
    parser.add_argument('--layer_name', type=str, default='model.layer1[0].conv3', help='Specify the layer name')
    parser.add_argument('--expansion_factor', type=int, default=2, help='Specify the expansion factor')
    parser.add_argument('--directory_path', type=str, default=r'C:\Users\Jasper\Downloads\Master thesis\Code', help='Specify the directory path')
    parser.add_argument('--metrics', nargs='+', default=['ce', 'percentage_same_classification', 'print_classifications', 'intermediate_feature_maps_similarity'], help='Specify the metrics to print')
    parser.add_argument('--model_epochs', type=int, default=5, help='Specify the model epochs')
    parser.add_argument('--model_learning_rate', type=float, default=0.1, help='Specify the model learning rate')
    parser.add_argument('--model_optimizer', type=str, default='sgd', help='Specify the model optimizer')
    parser.add_argument('--sae_epochs', type=int, default=2, help='Specify the sae epochs')
    parser.add_argument('--sae_learning_rate', type=float, default=0.001, help='Specify the sae learning rate')
    parser.add_argument('--sae_optimizer', type=str, default='adam', help='Specify the sae optimizer')

    # These 5 arguments are False by default. If they are specified on the command line, they will be True due
    # due to action='store_true'.
    parser.add_argument('--train_model', action='store_true', default=False, help='Train model')
    parser.add_argument('--evaluate_model', action='store_true', default=False, help='Evaluate model')
    parser.add_argument('--store_activations', action='store_true', default=False, help='Store activations')
    parser.add_argument('--train_sae', action='store_true', default=False, help='Train SAE')
    parser.add_argument('--modify_and_store_activations', action='store_true', default=False, help='Modify and store activations')
    parser.add_argument('--evaluate_modified_model', action='store_true', default=False, help='Evaluate modified model')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()

    # If all of them are False, then they are set to True, so that we don't have to specify all
    # of them if we want to use all of them
    if args.train_model == False and args.store_activations == False and args.train_sae == False and args.modify_and_store_activations == False and args.evaluate_model == False and args.evaluate_modified_model == False:
        args.train_model = True
        args.store_activations = True
        args.train_sae = True
        args.modify_and_store_activations = True
        args.evaluate_model = True
        args.evaluate_modified_model = True

    model_name = args.model_name
    sae_model_name = args.sae_model_name
    dataset_name = args.dataset_name
    layer_name = args.layer_name
    expansion_factor = args.expansion_factor
    directory_path = args.directory_path
    metrics = args.metrics
    model_epochs = args.model_epochs
    model_learning_rate = args.model_learning_rate
    model_optimizer = args.model_optimizer
    sae_epochs = args.sae_epochs
    sae_learning_rate = args.sae_learning_rate
    sae_optimizer = args.sae_optimizer

    original_activations_folder_path = os.path.join(directory_path, 'original_feature_maps', model_name, dataset_name)
    model_weights_folder_path = os.path.join(directory_path, 'model_weights', model_name, dataset_name)
    sae_weights_folder_path = os.path.join(directory_path, 'trained_sae_weights', model_name, dataset_name)
    adjusted_activations_folder_path = os.path.join(directory_path, 'adjusted_feature_maps', model_name, dataset_name)
    
    # Step 1: Train model 
    if args.train_model:
        if model_name == 'resnet50':
            pass # since resnet50 is pretrained
        elif model_name == 'custom_mlp_1':         
            # MAKE MODEL EPOCHS AND LEARNING RATE PARAMETERS LATER
            train_pipeline = TrainingPipeline(model_name,
                                            dataset_name,
                                            epochs=model_epochs,
                                            learning_rate=model_learning_rate, 
                                            batch_size=32,
                                            weights_folder_path=model_weights_folder_path)
            train_pipeline.execute_training(criterion_name='cross_entropy', 
                                            optimizer_name=model_optimizer)   #'adam' 'sgd'
            train_pipeline.save_model_weights()  

    # Step 1.1: Evaluate model
    if args.evaluate_model:
        model_evaluator = ModelEvaluator('single_model',
                                         model_name,
                                         dataset_name,
                                         batch_size=32,
                                         weights_folder_path = model_weights_folder_path)
        model_evaluator.evaluate()       

    # Step 2: Store Activations
    if args.store_activations:
        activations_handler = ActivationsHandler(model_name=model_name, 
                                                layer_name=layer_name, 
                                                dataset_name=dataset_name, 
                                                original_activations_folder_path=original_activations_folder_path, 
                                                sae_weights_folder_path=sae_weights_folder_path, 
                                                modify_output=False,
                                                batch_size=32)
        activations_handler.forward_pass()

    # MAKE BATCH SIZE A PARAMETER! PAY ATTENTION WHERE I NEED DIFFERENT BATCH SIZES!!!

    # Step 3: Train SAE on Stored Activations
    if args.train_sae:
        train_pipeline_sae = TrainingPipeline(model_name=sae_model_name,
                                            dataset_name='intermediate_feature_maps',
                                            data_dir=original_activations_folder_path, 
                                            layer_name=layer_name,
                                            epochs=sae_epochs,
                                            learning_rate=sae_learning_rate,
                                            batch_size=32,
                                            weights_folder_path=sae_weights_folder_path,
                                            expansion_factor=expansion_factor)
        train_pipeline_sae.execute_training(criterion_name='sae_loss',
                                            optimizer_name=sae_optimizer,
                                            lambda_sparse=0.1)
        train_pipeline_sae.save_model_weights()

    # Step 4: 
    # - modify output of layer "layer_name" with trained SAE using a hook
    # - evaluate the model on this adjusted feature map 
    # - store activations of this modified model
    if args.modify_and_store_activations:
        activations_handler_2 = ActivationsHandler(model_name = model_name, 
                                                layer_name = layer_name, 
                                                dataset_name = dataset_name,
                                                original_activations_folder_path=original_activations_folder_path, 
                                                sae_weights_folder_path=sae_weights_folder_path,
                                                adjusted_activations_folder_path=adjusted_activations_folder_path,
                                                sae_model_name=sae_model_name,
                                                sae_dataset_name='intermediate_feature_maps',
                                                modify_output=True, 
                                                expansion_factor=expansion_factor,
                                                batch_size=32) # batch size here should be the same as in previous activations handler object
        activations_handler_2.forward_pass()
    
    # Step 5: Evaluate how "similar" the modified model is to the original model
    if args.evaluate_modified_model:
        model_evaluator = ModelEvaluator('compare_models',
                                         model_name,
                                        dataset_name,
                                        layer_name, 
                                        batch_size=32,	
                                        original_activations_folder_path=original_activations_folder_path, 
                                        adjusted_activations_folder_path=adjusted_activations_folder_path,)
        model_evaluator.evaluate(metrics = metrics)