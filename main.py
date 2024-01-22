import os
import argparse

from activations_handler import ActivationsHandler
from train_pipeline import TrainingPipeline
from evaluate_modified_model import ModifiedModelEvaluator
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
    parser.add_argument('--dataset_name', type=str, default='tiny_imagenet', help='Specify the dataset name')
    parser.add_argument('--layer_name', type=str, default='model.layer1[0].conv3', help='Specify the layer name')
    parser.add_argument('--expansion_factor', type=int, default=2, help='Specify the expansion factor')
    parser.add_argument('--directory_path', type=str, default=r'C:\Users\Jasper\Downloads\Master thesis\Code', help='Specify the directory path')
    parser.add_argument('--metrics', nargs='+', default=['ce', 'percentage_same_classification', 'print_classifications', 'intermediate_feature_maps_similarity'], help='Specify the metrics to print')
    # ADD
    # model train epochs, Default should be None
    # sae train epochs
    # model train learning rate, Default should be None
    # sae train learning rate

    # These 5 arguments are False by default. If they are specified on the command line, they will be True due
    # due to action='store_true'.
    parser.add_argument('--train_model', action='store_true', default=False, help='Train model')
    parser.add_argument('--store_activations', action='store_true', default=False, help='Store activations')
    parser.add_argument('--train_sae', action='store_true', default=False, help='Train SAE')
    parser.add_argument('--modify_and_store_activations', action='store_true', default=False, help='Modify and store activations')
    parser.add_argument('--evaluate_model', action='store_true', default=False, help='Evaluate model')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()

    # If all of them are False, then they are set to True, so that we don't have to specify all
    # of them if we want to use all of them
    if args.train_model == False and args.store_activations == False and args.train_sae == False and args.modify_and_store_activations == False and args.evaluate_model == False:
        args.train_model = True
        args.store_activations = True
        args.train_sae = True
        args.modify_and_store_activations = True
        args.evaluate_model = True

    model_name = args.model_name
    dataset_name = args.dataset_name
    layer_name = args.layer_name
    expansion_factor = args.expansion_factor
    directory_path = args.directory_path
    metrics = args.metrics

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
                                       epochs=2, 
                                       learning_rate=0.001, 
                                       weights_folder_path=model_weights_folder_path)
            train_pipeline.execute_training(criterion_name='cross_entropy', 
                                            optimizer_name='sgd')   
            train_pipeline.save_model_weights()         

    # Step 2: Store Activations
    if args.store_activations:
        activations_handler = ActivationsHandler(model_name, 
                                                layer_name, 
                                                dataset_name, 
                                                original_activations_folder_path, 
                                                sae_weights_folder_path, 
                                                modify_output=False)
        activations_handler.register_hooks()    
        activations_handler.forward_pass()
        activations_handler.save_activations()

    # Step 3: Train SAE on Stored Activations
    if args.train_sae:
        # MAKE SAE EPOCHS AND LR PARAMETERS LATER
        train_pipeline_sae = TrainingPipeline(model_name='sae',
                                   dataset_name='intermediate_feature_maps',
                                   data_dir=original_activations_folder_path, 
                                   layer_name=layer_name,
                                   epochs=2,
                                   learning_rate=0.001,
                                   expansion_factor=expansion_factor,
                                   )
        train_pipeline_sae.execute_training(criterion_name='sae_loss',
                                            optimizer_name='sgd',
                                            lambda_sparse=0.1)
        train_pipeline_sae.save_model_weights()

    # Step 4: 
    # - modify output of layer "layer_name" with trained SAE using a hook
    # - evaluate the model on this adjusted feature map 
    # - store activations of this modified model
    if args.modify_and_store_activations:
        activations_handler = ActivationsHandler(model_name, 
                                                layer_name, 
                                                dataset_name, 
                                                adjusted_activations_folder_path, 
                                                sae_weights_folder_path,
                                                modify_output=True, 
                                                expansion_factor=expansion_factor)
        activations_handler.register_hooks()
        activations_handler.forward_pass()
        activations_handler.save_activations() 
    
    # Step 5: Evaluate how "similar" the modified model is to the original model
    if args.evaluate_model:
        model_evaluator = ModelEvaluator(model_name,
                                        dataset_name,
                                        layer_name, 
                                        original_activations_folder_path, 
                                        adjusted_activations_folder_path)
        model_evaluator.evaluate(metrics = metrics)