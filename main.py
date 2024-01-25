import os
import argparse

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
    parser.add_argument('--metrics', nargs='+', default=['ce', 'percentage_same_classification', 'intermediate_feature_maps_similarity', 'train_accuracy', 'visualize_classifications'], help='Specify the metrics to print')
    parser.add_argument('--model_epochs', type=int, default=5, help='Specify the model epochs')
    parser.add_argument('--model_learning_rate', type=float, default=0.1, help='Specify the model learning rate')
    parser.add_argument('--model_optimizer', type=str, default='sgd', help='Specify the model optimizer')
    parser.add_argument('--sae_epochs', type=int, default=2, help='Specify the sae epochs')
    parser.add_argument('--sae_learning_rate', type=float, default=0.001, help='Specify the sae learning rate')
    parser.add_argument('--sae_optimizer', type=str, default='adam', help='Specify the sae optimizer')
    parser.add_argument('--batch_size', type=int, default=32, help='Specify the batch size')
    parser.add_argument('--sae_batch_size', type=int, default=32, help='Specify the batch size for the feature maps')

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

    original_activations_folder_path = os.path.join(directory_path, 'original_feature_maps', model_name, dataset_name)
    model_weights_folder_path = os.path.join(directory_path, 'model_weights', model_name, dataset_name)
    sae_weights_folder_path = os.path.join(directory_path, 'model_weights', sae_model_name, dataset_name)
    adjusted_activations_folder_path = os.path.join(directory_path, 'adjusted_feature_maps', model_name, dataset_name)
    
    # Step 0: Load data loader (so that in each step we use the same order of batches)
    train_dataloader, val_dataloader, category_names = load_data(dataset_name, batch_size)
    img_size = get_img_size(dataset_name)

    # Step 1: Train model 
    if args.train_model:
        if model_name == 'resnet50':
            pass # since resnet50 is pretrained
        elif model_name == 'custom_mlp_1':  
            model = load_model(model_name, img_size)
            training = Training(model=model,
                                optimizer_name=model_optimizer,
                                criterion_name='cross_entropy',
                                learning_rate=model_learning_rate)
            training.train(train_dataloader=train_dataloader,
                           num_epochs=model_epochs,
                           valid_dataloader=val_dataloader)
            training.save_model(model_weights_folder_path)

    # Step 1.1: Evaluate model
    if args.evaluate_model:
        model = load_pretrained_model(model_name,
                                      img_size,
                                      model_weights_folder_path)
        print_model_accuracy(model, train_dataloader)
        show_classification_with_images(train_dataloader,category_names,model=model)
                                    
    # Step 2: Store Activations
    if args.store_activations:
        model = load_model(model_name, img_size)
        model.eval()
        activations_handler = ActivationsHandler(model = model, 
                                                train_dataloader=train_dataloader,
                                                layer_name = layer_name, 
                                                dataset_name = dataset_name,
                                                original_activations_folder_path=original_activations_folder_path, 
                                                adjusted_activations_folder_path=adjusted_activations_folder_path)
        activations_handler.forward_pass()
        activations_handler.save_activations()

    # Step 3: Train SAE on Stored Activations
    if args.train_sae:
        feature_maps_dataset = IntermediateActivationsDataset(layer_name=layer_name, 
                                                            root_dir=original_activations_folder_path, 
                                                            batch_size=sae_batch_size)
        sae_train_dataloader = DataLoader(feature_maps_dataset, 
                                          batch_size=sae_batch_size, 
                                          shuffle=True)
        sae_img_size = feature_maps_dataset.get_image_size()
        sae_val_dataloader = None
        sae_model = load_model(sae_model_name, sae_img_size, sae_expansion_factor)
        training_sae = Training(model=sae_model,
                                optimizer_name=sae_optimizer,
                                criterion_name='sae_loss',
                                learning_rate=sae_learning_rate,
                                lambda_sparse=0.1)
        training.train(train_dataloader=sae_train_dataloader,
                       num_epochs=sae_epochs,
                       valid_dataloader=sae_val_dataloader)
        training.save_model(sae_weights_folder_path, layer_name)

    # Step 4: 
    # - modify output of layer "layer_name" with trained SAE using a hook
    # - evaluate the model on this adjusted feature map 
    # - store activations of this modified model
    if args.modify_and_store_activations:  
        sae_model = load_pretrained_model(sae_model_name,
                                        sae_img_size,
                                        sae_weights_folder_path,
                                        sae_expansion_factor,
                                        layer_name)
        model = load_model(model_name, img_size, sae_expansion_factor)
        model.eval()
        activations_handler_modify = ActivationsHandler(model = model, 
                                                        train_dataloader=train_dataloader,
                                                        layer_name = layer_name, 
                                                        dataset_name = dataset_name,
                                                        original_activations_folder_path=original_activations_folder_path, 
                                                        adjusted_activations_folder_path=adjusted_activations_folder_path,
                                                        sae_model=sae_model)
        activations_handler_modify.forward_pass()
        activations_handler_modify.save_activations()
    
    # Step 5: Evaluate how "similar" the modified model is to the original model
    if args.evaluate_modified_model:
        model = load_pretrained_model(model_name,
                                      img_size,
                                      model_weights_folder_path)
        evaluate_feature_maps(original_activations_folder_path,
                              adjusted_activations_folder_path,
                              class_names=category_names,
                              metrics=metrics,
                              model = model, 
                              train_dataloader=train_dataloader) 