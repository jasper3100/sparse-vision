import os
import argparse
import time
import torch
import wandb

from activations_handler import ActivationsHandler
from training import Training
#from utils import get_img_size, load_data, load_pretrained_model, load_model, get_model_accuracy, log_image_table, show_classification_with_images, get_file_path
from utils import *
from dataloaders.intermediate_feature_map_dataset import IntermediateActivationsDataset
from torch.utils.data import DataLoader
from evaluate_feature_maps import evaluate_feature_maps
# I can run main.py as in the line below (or if I leave the arguments empty, it will use the default values)
# python main.py --store_activations --train_sae --modify_and_store_activations --model_name resnet18 --dataset_name cifar10 --layer_names model.layer2[0].conv1 --sae_expansion_factor 3 --directory_path C:\Users\Jasper\Downloads\Master thesis\Code
# python main.py --model_name resnet50 --dataset_name tiny_imagenet --layer_names model.layer1[0].conv3 --sae_expansion_factor 2 --directory_path C:\Users\Jasper\Downloads\Master thesis\Code --metrics ce percentage_same_classification

def parse_arguments(name=None):
    parser = argparse.ArgumentParser(description="Setting parameters")

    # command-line arguments
    parser.add_argument('--model_name', type=str, default='resnet50', help='Specify the model name')
    parser.add_argument('--sae_model_name', type=str, default='sae_mlp', help='Specify the sae model name')
    parser.add_argument('--dataset_name', type=str, default='tiny_imagenet', help='Specify the dataset name')
    parser.add_argument('--layer_names', nargs='+', type=str, default=['model.layer1[0].conv3'], help='Specify the layer names')
    # Example command: python main.py --layer_names model.layer1[0].conv3 model.layer1[0].conv2
    parser.add_argument('--directory_path', type=str, default=r'C:\Users\Jasper\Downloads\Master thesis\Code', help='Specify the directory path')
    #parser.add_argument('--metrics', nargs='+', default=['kld', 'percentage_same_classification', 'intermediate_feature_maps_similarity', 'train_accuracy', 'visualize_classifications', 'sparsity', 'polysemanticity_level'], help='Specify the metrics to print')
    parser.add_argument('--run_group_ID', type=str, default='main', help='ID of group of runs for W&B')
    parser.add_argument('--wandb_status', type=str, default='True', help='Specify whether to use W&B')

    # if we are in main.py, and not in main_gridsearch.py
    if name=='main':
        parser.add_argument('--steps_to_execute', type=str, default='1234567', help='Specify which steps to execute')
        parser.add_argument('--model_epochs', type=int, default=5, help='Specify the model epochs')
        parser.add_argument('--model_learning_rate', type=float, default=0.1, help='Specify the model learning rate')
        parser.add_argument('--model_optimizer', type=str, default='sgd', help='Specify the model optimizer')
        parser.add_argument('--sae_epochs', type=int, default=5, help='Specify the sae epochs')
        parser.add_argument('--sae_learning_rate', type=float, default=0.001, help='Specify the sae learning rate')
        parser.add_argument('--sae_optimizer', type=str, default='adam', help='Specify the sae optimizer')
        parser.add_argument('--batch_size', type=int, default=32, help='Specify the batch size')
        parser.add_argument('--sae_batch_size', type=int, default=32, help='Specify the batch size for the feature maps')
        parser.add_argument('--sae_lambda_sparse', type=float, default=0.1, help='Specify the lambda sparse')
        parser.add_argument('--sae_expansion_factor', type=int, default=2, help='Specify the expansion factor')
        parser.add_argument('--activation_threshold', type=float, default=0.1, help='Specify the activation threshold')
        # NEED TO DECIDE SOMEHOW WHEN A NEURON WOULD BE COUNTED AS ACTIVATED... MAYBE RELATIVE TO THE MAGNITUDE OF ALL ACTIVATIONS???
    elif name=='gridsearch':
        pass
    else:
        raise ValueError('Specify whether you are in main.py or in main_gridsearch.py')
    return parser.parse_args()

def get_vars(args, name):
    if name=='main':
        return args.model_name, args.sae_model_name, args.dataset_name, args.layer_names, args.directory_path, args.run_group_ID, eval(args.wandb_status), args.steps_to_execute, args.model_epochs, args.model_learning_rate, args.model_optimizer, args.sae_epochs, args.sae_learning_rate, args.sae_optimizer, args.batch_size, args.sae_batch_size, args.sae_lambda_sparse, args.sae_expansion_factor, args.activation_threshold
    elif name=='gridsearch':
        return args.model_name, args.sae_model_name, args.dataset_name, args.layer_names, args.directory_path, args.run_group_ID, eval(args.wandb_status)

def execute_project(model_name, 
                    sae_model_name, 
                    dataset_name, 
                    layer_names, 
                    directory_path, 
                    #metrics, 
                    run_group_ID,
                    wandb_status,
                    steps_to_execute, 
                    model_epochs, 
                    model_learning_rate, 
                    model_optimizer, 
                    sae_epochs, 
                    sae_learning_rate, 
                    sae_optimizer, 
                    batch_size, 
                    sae_batch_size,
                    sae_lambda_sparse,
                    sae_expansion_factor,
                    activation_threshold):
    # These parameter dictionaries are used for creating file names, f.e., to store model weights, feature maps, etc. Hence, include any parameter here that you would like to 
    # be included in the file names to better use and identify files, model_name and dataset_name are already considered
    model_params = {'epochs': model_epochs, 'learning_rate': model_learning_rate, 'batch_size': batch_size, 'optimizer': model_optimizer, 'activation_threshold': activation_threshold}
    sae_params = {'epochs': sae_epochs, 'learning_rate': sae_learning_rate, 'batch_size': sae_batch_size, 'optimizer': sae_optimizer, 'expansion_factor': sae_expansion_factor, 'lambda_sparse': sae_lambda_sparse, 'activation_threshold': activation_threshold}

    original_activations_folder_path = os.path.join(directory_path, 'original_feature_maps', model_name, dataset_name)
    model_weights_folder_path = os.path.join(directory_path, 'model_weights', model_name, dataset_name)
    sae_weights_folder_path = os.path.join(directory_path, 'model_weights', sae_model_name, dataset_name)
    adjusted_activations_folder_path = os.path.join(directory_path, 'adjusted_feature_maps', model_name, dataset_name)
    evaluation_results_folder_path = os.path.join(directory_path, 'evaluation_results', model_name, dataset_name)
    encoder_output_folder_path = os.path.join(directory_path, 'encoder_output', model_name, dataset_name)
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('Using GPU')
    else:
        device = torch.device('cpu')
        print('Using CPU')

    run_group_ID = steps_to_execute + "_" + run_group_ID
    run_ID = get_file_path(layer_names=layer_names, params=model_params, params2=sae_params, file_name=run_group_ID)

    # steps 3 and 6 (storing feature maps do not require logging to W&B)
    if wandb_status and any(x in steps_to_execute for x in ['1', '2', '4', '5', '7']):
        wandb.login()
        wandb.init(project="master-thesis",
                    name=run_ID,
                    group=run_group_ID,
                    #job_type="train", can specify job type for adding description
                    config={"run_ID": run_ID,
                            "run_group_ID": run_group_ID,
                            "steps_to_execute": steps_to_execute,
                            "model_name": model_name,
                            "sae_model_name": sae_model_name,
                            "dataset_name": dataset_name,
                            "layer_names": layer_names,
                            "sae_expansion_factor": sae_expansion_factor,
                            "directory_path": directory_path,
                            #"metrics": metrics,
                            "model_epochs": model_epochs,
                            "model_learning_rate": model_learning_rate,
                            "model_optimizer": model_optimizer,
                            "sae_epochs": sae_epochs,
                            "sae_learning_rate": sae_learning_rate,
                            "sae_optimizer": sae_optimizer,
                            "batch_size": batch_size,
                            "sae_batch_size": sae_batch_size,
                            "sae_lambda_sparse": sae_lambda_sparse,
                            "activation_threshold": activation_threshold},)

    # Step 0: Load data loader (so that when evaluating the output feature maps later on, they are in the same order
    # that was used to train the model in the first place)
    train_dataloader, val_dataloader, category_names, train_dataset_length = load_data(directory_path, dataset_name, batch_size)
    num_classes = len(category_names)
    img_size = get_img_size(dataset_name)

    # Step 1: Train model 
    if "1" in steps_to_execute:
        if model_name == 'resnet50':
            pass # since resnet50 is pretrained
        elif model_name == 'custom_mlp_1':  
            model = load_model(model_name, img_size)
            model = model.to(device)
            #with torch.autograd.profiler.profile(use_cuda=True) as prof:
            training = Training(model=model,
                                model_name=model_name,
                                device=device,
                                optimizer_name=model_optimizer,
                                criterion_name='cross_entropy',
                                learning_rate=model_learning_rate)
            training.train(train_dataloader=train_dataloader,
                            num_epochs=model_epochs,
                            wandb_status=wandb_status,
                            valid_dataloader=val_dataloader)
            #print(prof.key_averages().table(sort_by="cuda_time_total"))
            training.save_model(model_weights_folder_path, params=model_params)
        
    # Step 2: Evaluate model
    if "2" in steps_to_execute:
        model = load_pretrained_model(model_name,
                                    img_size,
                                    model_weights_folder_path,
                                    params=model_params)
        model = model.to(device)
        get_model_accuracy(model, 
                           device, 
                           train_dataloader, 
                           original_activations_folder_path=original_activations_folder_path,
                           layer_names=layer_names,
                           model_params=model_params,
                           wandb_status=wandb_status)
        if wandb_status:
            log_image_table(train_dataloader,
                category_names,
                model=model, 
                device=device)
        else:
            show_classification_with_images(train_dataloader,
                                            category_names,
                                            folder_path=evaluation_results_folder_path,
                                            model=model, 
                                            device=device,
                                            params=model_params)

        
    # Step 3: Store Activations
    if "3" in steps_to_execute:
        model = load_pretrained_model(model_name,
                                      img_size,
                                      model_weights_folder_path,
                                      params=model_params)
        model = model.to(device)
        activations_handler = ActivationsHandler(model = model, 
                                                 device=device,
                                                train_dataloader=train_dataloader,
                                                layer_names = layer_names, 
                                                dataset_name = dataset_name,
                                                folder_path=original_activations_folder_path,
                                                activation_threshold=activation_threshold,
                                                params=model_params) 
        activations_handler.forward_pass()
        activations_handler.save_activations()
        
    # Step 4: Train SAE on Stored Activations
    if "4" in steps_to_execute:
        feature_maps_dataset = IntermediateActivationsDataset(layer_names=layer_names, 
                                                            original_activations_folder_path=original_activations_folder_path,
                                                            train_dataset_length=train_dataset_length,
                                                            params=model_params)
        sae_train_dataloader = DataLoader(feature_maps_dataset, 
                                          batch_size=sae_batch_size, 
                                          #num_workers=4,
                                          shuffle=True)
        sae_img_size = feature_maps_dataset.get_image_size()
        sae_val_dataloader = None
        sae_model = load_model(sae_model_name, sae_img_size, sae_expansion_factor)
        sae_model = sae_model.to(device)
        # we create another train_dataloader with sae_batch_size so that we can compute the polysemanticity level 
        train_dataloader_2, _, _, _ = load_data(directory_path, dataset_name, sae_batch_size)
        # We ensure (based on the labels of the first batch) that train_dataloader and train_dataloader_2 are in the same order
        if not torch.equal(next(iter(train_dataloader))[1], next(iter(train_dataloader_2))[1]):
            raise ValueError("The labels of the first batch of train_dataloader and train_dataloader_2 are not the same")
        #'''
        #with torch.autograd.profiler.profile(use_cuda=True) as prof:
        training_sae = Training(model=sae_model,
                                model_name=sae_model_name,
                                device=device,
                                optimizer_name=sae_optimizer,
                                criterion_name='sae_loss',
                                learning_rate=sae_learning_rate,
                                lambda_sparse=sae_lambda_sparse,
                                dataloader_2=train_dataloader,
                                num_classes=num_classes,
                                activation_threshold=activation_threshold,
                                expansion_factor=sae_expansion_factor)
        training_sae.train(train_dataloader=sae_train_dataloader,
                            num_epochs=sae_epochs,
                            wandb_status=wandb_status,
                            valid_dataloader=sae_val_dataloader,
                            train_dataset_length=train_dataset_length,
                            folder_path=evaluation_results_folder_path,
                            layer_names=layer_names,
                            params=sae_params)
        #print(prof.key_averages().table(sort_by="cuda_time_total"))
        training_sae.save_model(sae_weights_folder_path, layer_names=layer_names, params=sae_params)
        #'''

    # Step 5: Evaluate SAE
    if "5" in steps_to_execute:
        get_sae_losses(evaluation_results_folder_path, layer_names, sae_params, wandb_status)
        
    # Step 6: 
    # - modify output of layer "layer_names" with trained SAE using a hook
    # - evaluate the model on this adjusted feature map 
    # - store activations of this modified model
    if "6" in steps_to_execute:
        start4 = time.process_time()
        # we instantiate this dataset here only for getting sae_img_size
        feature_maps_dataset = IntermediateActivationsDataset(layer_names=layer_names, 
                                                                original_activations_folder_path=original_activations_folder_path, 
                                                                train_dataset_length=train_dataset_length,
                                                                params=model_params)
        sae_img_size = feature_maps_dataset.get_image_size()
        sae_model = load_pretrained_model(sae_model_name,
                                        sae_img_size,
                                        sae_weights_folder_path,
                                        sae_expansion_factor=sae_expansion_factor,
                                        layer_names=layer_names,
                                        params=sae_params)
        sae_model = sae_model.to(device)
        model = load_pretrained_model(model_name,
                                      img_size,
                                      model_weights_folder_path,
                                      params=model_params)
        model = model.to(device)
        activations_handler_modify = ActivationsHandler(model = model, 
                                                        device=device,
                                                        train_dataloader=train_dataloader,
                                                        layer_names = layer_names, 
                                                        dataset_name = dataset_name,
                                                        folder_path=adjusted_activations_folder_path,
                                                        activation_threshold=activation_threshold,
                                                        sae_model=sae_model,
                                                        params=sae_params,
                                                        encoder_output_folder_path=encoder_output_folder_path)
        activations_handler_modify.forward_pass()
        activations_handler_modify.save_activations()
        print("Seconds taken to modify and store activations: ", time.process_time() - start4)
    
    # Step 7: Evaluate how "similar" the modified model is to the original model
    if "7" in steps_to_execute:
        start5 = time.process_time()
        model = load_pretrained_model(model_name,
                                      img_size,
                                      model_weights_folder_path,
                                      params=model_params)
        model = model.to(device)
        evaluate_feature_maps(original_activations_folder_path,
                              adjusted_activations_folder_path,
                              model_params,
                              sae_params,
                              wandb_status,
                              evaluation_results_folder_path=evaluation_results_folder_path,
                              class_names=category_names,
                              #metrics=metrics,
                              model=model, 
                              device=device,
                              train_dataloader=train_dataloader,
                              layer_names=layer_names,
                              train_dataset_length=train_dataset_length,
                              encoder_output_folder_path=encoder_output_folder_path,
                              activation_threshold=activation_threshold,
                              num_classes=num_classes) 
        print("Seconds taken to evaluate modified model: ", time.process_time() - start5)
    
    wandb.finish()

if __name__ == '__main__':
    args = parse_arguments(name='main')
    variables = get_vars(args, name='main')
    execute_project(*variables)