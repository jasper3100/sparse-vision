import os
import argparse
import time
import torch

from activations_handler import ActivationsHandler
from training import Training
from utils import get_img_size, load_data, load_pretrained_model, load_model, print_model_accuracy, show_classification_with_images
from dataloaders.intermediate_feature_map_dataset import IntermediateActivationsDataset
from torch.utils.data import DataLoader
from evaluate_feature_maps import evaluate_feature_maps
# I can run main.py as in the line below (or if I leave the arguments empty, it will use the default values)
# python main.py --store_activations --train_sae --modify_and_store_activations --model_name resnet18 --dataset_name cifar10 --layer_name model.layer2[0].conv1 --sae_expansion_factor 3 --directory_path C:\Users\Jasper\Downloads\Master thesis\Code
# python main.py --model_name resnet50 --dataset_name tiny_imagenet --layer_name model.layer1[0].conv3 --sae_expansion_factor 2 --directory_path C:\Users\Jasper\Downloads\Master thesis\Code --metrics ce percentage_same_classification

def parse_arguments():
    parser = argparse.ArgumentParser(description="Setting parameters")

    # command-line arguments
    parser.add_argument('--model_name', type=str, default='resnet50', help='Specify the model name')
    parser.add_argument('--sae_model_name', type=str, default='sae_mlp', help='Specify the sae model name')
    parser.add_argument('--dataset_name', type=str, default='tiny_imagenet', help='Specify the dataset name')
    parser.add_argument('--layer_name', type=str, default='model.layer1[0].conv3', help='Specify the layer name')
    parser.add_argument('--sae_expansion_factor', type=int, default=2, help='Specify the expansion factor')
    parser.add_argument('--directory_path', type=str, default=r'C:\Users\Jasper\Downloads\Master thesis\Code', help='Specify the directory path')
    parser.add_argument('--metrics', nargs='+', default=['kld', 'percentage_same_classification', 'intermediate_feature_maps_similarity', 'train_accuracy', 'visualize_classifications', 'sparsity'], help='Specify the metrics to print')
    parser.add_argument('--model_epochs', type=int, default=5, help='Specify the model epochs')
    parser.add_argument('--model_learning_rate', type=float, default=0.1, help='Specify the model learning rate')
    parser.add_argument('--model_optimizer', type=str, default='sgd', help='Specify the model optimizer')
    parser.add_argument('--sae_epochs', type=int, default=5, help='Specify the sae epochs')
    parser.add_argument('--sae_learning_rate', type=float, default=0.001, help='Specify the sae learning rate')
    parser.add_argument('--sae_optimizer', type=str, default='adam', help='Specify the sae optimizer')
    parser.add_argument('--batch_size', type=int, default=32, help='Specify the batch size')
    parser.add_argument('--sae_batch_size', type=int, default=32, help='Specify the batch size for the feature maps')
    parser.add_argument('--eval_sparsity_threshold', type=float, default=0.05, help='Specify the sparsity threshold')

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
    sae_expansion_factor = args.sae_expansion_factor
    directory_path = args.directory_path
    metrics = args.metrics
    model_epochs = args.model_epochs
    model_learning_rate = args.model_learning_rate
    model_optimizer = args.model_optimizer
    sae_epochs = args.sae_epochs
    sae_learning_rate = args.sae_learning_rate
    sae_optimizer = args.sae_optimizer
    batch_size = args.batch_size
    sae_batch_size = args.sae_batch_size
    eval_sparsity_threshold = args.eval_sparsity_threshold
    model_params = {'epochs': model_epochs, 'learning_rate': model_learning_rate, 'batch_size': batch_size, 'optimizer': model_optimizer}
    sae_params = {'epochs': sae_epochs, 'learning_rate': sae_learning_rate, 'batch_size': sae_batch_size, 'optimizer': sae_optimizer}

    original_activations_folder_path = os.path.join(directory_path, 'original_feature_maps', model_name, dataset_name)
    model_weights_folder_path = os.path.join(directory_path, 'model_weights', model_name, dataset_name)
    sae_weights_folder_path = os.path.join(directory_path, 'model_weights', sae_model_name, dataset_name)
    adjusted_activations_folder_path = os.path.join(directory_path, 'adjusted_feature_maps', model_name, dataset_name)
    evaluation_results_folder_path = os.path.join(directory_path, 'evaluation_results', model_name, dataset_name)
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('Using GPU')
    else:
        device = torch.device('cpu')
        print('Using CPU')

    start0 = time.process_time()
    # Step 0: Load data loader (so that when evaluating the output feature maps later on, they are in the same order
    # that was used to train the model in the first place)
    train_dataloader, val_dataloader, category_names, train_dataset_length = load_data(directory_path, dataset_name, batch_size)
    img_size = get_img_size(dataset_name)
    print("Seconds taken to load data: ", time.process_time() - start0)

    # Step 1: Train model 
    if args.train_model:
        #start1 = time.process_time()
        if model_name == 'resnet50':
            pass # since resnet50 is pretrained
        elif model_name == 'custom_mlp_1':  
            model = load_model(model_name, img_size)
            model = model.to(device)
            #with torch.autograd.profiler.profile(use_cuda=True) as prof:
            training = Training(model=model,
                                device=device,
                                optimizer_name=model_optimizer,
                                criterion_name='cross_entropy',
                                learning_rate=model_learning_rate)
            training.train(train_dataloader=train_dataloader,
                            num_epochs=model_epochs,
                            valid_dataloader=val_dataloader)
            #print(prof)
            #print(prof.key_averages().table(sort_by="cuda_time_total"))
            training.save_model(model_weights_folder_path, params=model_params)
        #print("Seconds taken to train model: ", time.process_time() - start1)

    # Step 1.1: Evaluate model
    if args.evaluate_model:
        #start11 = time.process_time()
        model = load_pretrained_model(model_name,
                                    img_size,
                                    model_weights_folder_path,
                                    params=model_params)
        model = model.to(device)
        print_model_accuracy(model, device, train_dataloader)
        show_classification_with_images(train_dataloader,
                                        category_names,
                                        folder_path=evaluation_results_folder_path,
                                        model=model, 
                                        device=device,
                                        params=model_params)
        #print("Seconds taken to evaluate model: ", time.process_time() - start11)

    # Step 2: Store Activations
    if args.store_activations:
        start2 = time.process_time()
        model = load_pretrained_model(model_name,
                                      img_size,
                                      model_weights_folder_path,
                                      params=model_params)
        model = model.to(device)
        activations_handler = ActivationsHandler(model = model, 
                                                 device=device,
                                                train_dataloader=train_dataloader,
                                                layer_name = layer_name, 
                                                dataset_name = dataset_name,
                                                folder_path=original_activations_folder_path,
                                                eval_sparsity_threshold=eval_sparsity_threshold,
                                                params=model_params) 
        activations_handler.forward_pass()
        activations_handler.save_activations()
        print("Seconds taken to store activations: ", time.process_time() - start2)

    # Step 3: Train SAE on Stored Activations
    if args.train_sae:
        #start3 = time.process_time()
        feature_maps_dataset = IntermediateActivationsDataset(layer_name=layer_name, 
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
        #print("Seconds taken to train SAE: ", time.time() - start3)
        #'''
        #with torch.autograd.profiler.profile(use_cuda=True) as prof:
        training_sae = Training(model=sae_model,
                                device=device,
                                optimizer_name=sae_optimizer,
                                criterion_name='sae_loss',
                                learning_rate=sae_learning_rate,
                                lambda_sparse=0.1)
        training_sae.train(train_dataloader=sae_train_dataloader,
                            num_epochs=sae_epochs,
                            valid_dataloader=sae_val_dataloader)
        #print(prof)
        #print(prof.key_averages().table(sort_by="cuda_time_total"))
        training_sae.save_model(sae_weights_folder_path, layer_name=layer_name, params=sae_params)
        #'''
        #print("Seconds taken to train SAE: ", time.process_time() - start3)

    # Step 4: 
    # - modify output of layer "layer_name" with trained SAE using a hook
    # - evaluate the model on this adjusted feature map 
    # - store activations of this modified model
    if args.modify_and_store_activations:
        start4 = time.process_time()
        # we instantiate this dataset here only for getting sae_img_size
        feature_maps_dataset = IntermediateActivationsDataset(layer_name=layer_name, 
                                                                original_activations_folder_path=original_activations_folder_path, 
                                                                train_dataset_length=train_dataset_length,
                                                                params=model_params)
        sae_img_size = feature_maps_dataset.get_image_size()
        sae_model = load_pretrained_model(sae_model_name,
                                        sae_img_size,
                                        sae_weights_folder_path,
                                        sae_expansion_factor=sae_expansion_factor,
                                        layer_name=layer_name,
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
                                                        layer_name = layer_name, 
                                                        dataset_name = dataset_name,
                                                        folder_path=adjusted_activations_folder_path,
                                                        eval_sparsity_threshold=eval_sparsity_threshold,
                                                        sae_model=sae_model,
                                                        params=sae_params)
        activations_handler_modify.forward_pass()
        activations_handler_modify.save_activations()
        print("Seconds taken to modify and store activations: ", time.process_time() - start4)
    
    # Step 5: Evaluate how "similar" the modified model is to the original model
    if args.evaluate_modified_model:
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
                              evaluation_results_folder_path=evaluation_results_folder_path,
                              class_names=category_names,
                              metrics=metrics,
                              model=model, 
                              device=device,
                              train_dataloader=train_dataloader,
                              layer_name=layer_name) 
        print("Seconds taken to evaluate modified model: ", time.process_time() - start5)