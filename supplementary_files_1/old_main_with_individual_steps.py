import os
import argparse
import time
import torch
import wandb
import random
import string
import ast

from activations_handler import ActivationsHandler
from training import Training
from utils import *
from dataloaders.intermediate_feature_map_dataset import IntermediateActivationsDataset
from torch.utils.data import DataLoader
from evaluate_feature_maps import evaluate_feature_maps
from model_pipeline import ModelPipeline

def parse_arguments():
    parser = argparse.ArgumentParser(description="Setting parameters")
    # command-line arguments
    parser.add_argument('--execution_location', type=str, help='Specify where to run the code')
    parser.add_argument('--model_name', type=str, help='Specify the model name')
    parser.add_argument('--sae_model_name', type=str, help='Specify the sae model name')
    parser.add_argument('--layer_names', type=str, help='Specify the layer names')
    parser.add_argument('--steps_to_execute', type=str, help='Specify which steps to execute')
    parser.add_argument('--directory_path', type=str, help='Specify the directory path')
    parser.add_argument('--wandb_status', type=str, help='Specify whether to use W&B')
    parser.add_argument('--model_epochs', type=int, help='Specify the model epochs')
    parser.add_argument('--model_learning_rate', type=float, help='Specify the model learning rate')
    parser.add_argument('--batch_size', type=int, help='Specify the batch size')
    parser.add_argument('--model_optimizer_name', type=str, help='Specify the model optimizer name')
    parser.add_argument('--sae_epochs', type=int, help='Specify the sae epochs')
    parser.add_argument('--sae_learning_rate', type=float, help='Specify the sae learning rate')
    parser.add_argument('--sae_optimizer_name', type=str, help='Specify the sae optimizer name')
    parser.add_argument('--sae_batch_size', type=int, help='Specify the batch size for the feature maps')
    parser.add_argument('--sae_lambda_sparse', type=float, help='Specify the lambda sparse')
    parser.add_argument('--sae_expansion_factor', type=int, help='Specify the expansion factor')
    parser.add_argument('--activation_threshold', type=float, help='Specify the activation threshold')
    parser.add_argument('--dataset_name', type=str, help='Specify the dataset name')
    parser.add_argument('--run_group_ID', type=str, help='ID of group of runs for W&B')
    return parser.parse_args()

def execute_project(model_name,
                    sae_model_name,
                    layer_names,
                    steps_to_execute,
                    directory_path,
                    wandb_status,
                    model_epochs,
                    model_learning_rate,
                    batch_size,
                    model_optimizer_name,
                    sae_epochs,
                    sae_learning_rate,
                    sae_optimizer_name,
                    sae_batch_size,
                    sae_lambda_sparse,
                    sae_expansion_factor,
                    activation_threshold,
                    dataset_name,
                    run_group_ID):
    wandb_status = eval(wandb_status) # Turn 'False' into False, 'True' into True
    layer_names = ast.literal_eval(layer_names) # turn the string ['fc1'] into an actual list

    # If we run the code locally and use the parameters specified in the txt file, we need to convert them
    # from string into the desired format
    model_epochs = int(model_epochs)
    model_learning_rate = float(model_learning_rate)
    batch_size = int(batch_size)
    sae_epochs = int(sae_epochs)
    sae_learning_rate = float(sae_learning_rate)
    sae_batch_size = int(sae_batch_size)
    sae_lambda_sparse = float(sae_lambda_sparse)
    sae_expansion_factor = float(sae_expansion_factor)
    activation_threshold = float(activation_threshold)

    # These parameter dictionaries are used for creating file names, f.e., to store model weights, feature maps, etc. Hence, include any parameter here that you would like to 
    # be included in the file names to better use and identify files, model_name and dataset_name are already considered
    model_params = {'epochs': model_epochs, 'learning_rate': model_learning_rate, 'batch_size': batch_size, 'optimizer': model_optimizer_name, 'activation_threshold': activation_threshold}
    sae_params = {'epochs': sae_epochs, 'learning_rate': sae_learning_rate, 'batch_size': sae_batch_size, 'optimizer': sae_optimizer_name, 'expansion_factor': sae_expansion_factor, 'lambda_sparse': sae_lambda_sparse, 'activation_threshold': activation_threshold}

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

    run_group_ID = steps_to_execute + "_" + dataset_name + "_" + run_group_ID
    run_ID = get_file_path(layer_names=layer_names, params=model_params, params2=sae_params, file_name=run_group_ID)

    # steps 3 and 6 (storing feature maps do not require logging to W&B)
    # REMOVE STEP 8 FROM HERE LATER
    if wandb_status and any(x in steps_to_execute for x in ['1', '2', '4', '5', '7', '8']):
        print("Logging to W&B")
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
                            "model_optimizer_name": model_optimizer_name,
                            "sae_epochs": sae_epochs,
                            "sae_learning_rate": sae_learning_rate,
                            "sae_optimizer": sae_optimizer_name,
                            "batch_size": batch_size,
                            "sae_batch_size": sae_batch_size,
                            "sae_lambda_sparse": sae_lambda_sparse,
                            "activation_threshold": activation_threshold})

    # Step 0: Load data loader once for all steps
    train_dataloader, val_dataloader, category_names, train_dataset_length, num_classes, num_batches, img_size = load_data(directory_path, dataset_name, batch_size)
    # num_batches can be set to a different value if we want to limit the number of batches (which can be used wherever desired)

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
                                optimizer_name=model_optimizer_name,
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
                            wandb_status=wandb_status,
                            num_batches=num_batches)
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
        '''
        def trace_handler(p):
            output = p.key_averages().table(sort_by="self_cuda_time_total", row_limit=10)
            print(output)
            p.export_chrome_trace("/tmp/trace_" + str(p.step_num) + ".json")
        #alternatively, torch.autograd.profiler.profile(...)
        with torch.profiler.profile(use_cuda=True,
                                                schedule=torch.profiler.schedule(skip_first=10,wait=5,warmup=1,active=2),
                                                on_trace_ready=trace_handler) as prof:
        # for detailed instructions, see: https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html
        '''
        #with torch.autograd.profiler.profile(use_cuda=True) as prof:
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
                                                params=model_params, 
                                                encoder_output_folder_path=encoder_output_folder_path,
                                                dataset_length=train_dataset_length) 
        activations_handler.forward_pass(num_batches=num_batches)
        activations_handler.save_activations()
        #print(prof.key_averages().table(sort_by="cuda_time_total"))
        
    # Step 8: Train SAE directly
    if "8" in steps_to_execute:
        use_sae = True
        train_sae = True
        train_original_model = False
        store_encoder_output = False
        store_modified_activations = False
        store_original_activations = False
        model_criterion_name = 'cross_entropy'
        sae_criterion_name = 'sae_loss'
        pipeline = ModelPipeline(device=device,
                                train_dataloader=train_dataloader,
                                layer_names=layer_names, 
                                activation_threshold=activation_threshold,
                                prof=None,
                                use_sae=use_sae,
                                train_sae=train_sae,
                                train_original_model=train_original_model,
                                store_encoder_output=store_encoder_output,
                                store_modified_activations=store_modified_activations,
                                store_original_activations=store_original_activations,
                                activations_folder_path=original_activations_folder_path,
                                sae_weights_folder_path=sae_weights_folder_path,
                                model_weights_folder_path=model_weights_folder_path)
        pipeline.instantiate_models(model_name=model_name, 
                                    img_size=img_size, 
                                    model_optimizer_name=model_optimizer_name,
                                    model_criterion_name=model_criterion_name,
                                    model_learning_rate=model_learning_rate,
                                    model_params=model_params,
                                    sae_model_name=sae_model_name,
                                    sae_expansion_factor=sae_expansion_factor,
                                    sae_lambda_sparse=sae_lambda_sparse,
                                    sae_optimizer_name=sae_optimizer_name,
                                    sae_criterion_name=sae_criterion_name,
                                    sae_learning_rate=sae_learning_rate,
                                    sae_params=sae_params)
        pipeline.deploy_model(num_epochs=model_epochs, 
                              wandb_status=wandb_status)
                                



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
        print(sae_img_size, type(sae_img_size))
        sae_model = load_model(sae_model_name, sae_img_size, sae_expansion_factor)
        sae_model = sae_model.to(device)
        # we create another train_dataloader with sae_batch_size so that we can compute the polysemanticity level 
        train_dataloader_2, _, _, _, _, _, _ = load_data(directory_path, dataset_name, sae_batch_size)
        # We ensure (based on the labels of the first batch) that train_dataloader and train_dataloader_2 are in the same order
        if not torch.equal(next(iter(train_dataloader))[1][:min(sae_batch_size, batch_size)], next(iter(train_dataloader_2))[1][:min(sae_batch_size, batch_size)]):
            print(next(iter(train_dataloader))[1][:min(sae_batch_size, batch_size)], next(iter(train_dataloader_2))[1][:min(sae_batch_size, batch_size)])
            raise ValueError("The labels of the first batch of train_dataloader and train_dataloader_2 are not the same")
        #'''
        #with torch.autograd.profiler.profile(use_cuda=True) as prof:
        training_sae = Training(model=sae_model,
                                model_name=sae_model_name,
                                device=device,
                                optimizer_name=sae_optimizer_name,
                                criterion_name='sae_loss',
                                learning_rate=sae_learning_rate,
                                lambda_sparse=sae_lambda_sparse,
                                dataloader_2=train_dataloader_2,
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
        #start4 = time.process_time()
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
        #print("Seconds taken to modify and store activations: ", time.process_time() - start4)

    # Step 7: Evaluate how "similar" the modified model is to the original model
    if "7" in steps_to_execute:
        #start5 = time.process_time()
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
                                num_classes=num_classes,
                                num_batches=num_batches) 
        #print("Seconds taken to evaluate modified model: ", time.process_time() - start5)

    wandb.finish()


if __name__ == '__main__':
    # Parse command line arguments
    args = parse_arguments()

    if args.execution_location is None or args.execution_location == 'local': 
        print("Run code locally")
        # create a random (and hopefully unique) group ID
        run_group_ID = "".join(random.choices(string.ascii_lowercase, k=10))

        # Read parameters from the file, line by line, and execute the code consecutively
        with open('parameters.txt', 'r') as file:
            for line in file:
                parameters = [param for param in line.strip().split(',')]
                execute_project(*parameters, run_group_ID)

    elif args.execution_location == 'cluster':
        print("Run code on cluster")
        execute_project(args.model_name,
                        args.sae_model_name,
                        args.layer_names,
                        args.steps_to_execute,
                        args.directory_path,
                        args.wandb_status,
                        args.model_epochs,
                        args.model_learning_rate,
                        args.batch_size,
                        args.model_optimizer_name,
                        args.sae_epochs,
                        args.sae_learning_rate,
                        args.sae_optimizer_name,
                        args.sae_batch_size,
                        args.sae_lambda_sparse,
                        args.sae_expansion_factor,
                        args.activation_threshold,
                        args.dataset_name,
                        args.run_group_ID)
    else:
        raise ValueError("Please specify a valid execution location: either 'local' or 'cluster' or None")