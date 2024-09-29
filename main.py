import argparse
import random
import string
import wandb

from execute_project import ExecuteProject
from utils import *

def parse_arguments():
    parser = argparse.ArgumentParser(description="Setting parameters")
    # command-line arguments
    parser.add_argument('--execution_location', type=str, help='Specify where to run the code')
    parser.add_argument('--run_pipeline', action='store_true', help='Run the model pipeline') 
    # action='store_true' means that if the flag is present, the value is True, otherwise False
    parser.add_argument('--run_evaluation', action='store_true', help='Run the evaluation')
    parser.add_argument('--model_name', type=str, help='Specify the model name')
    parser.add_argument('--sae_model_name', type=str, help='Specify the sae model name')
    parser.add_argument('--sae_layer', type=str, help='Specify the layer names')
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
    parser.add_argument('--dataset_name', type=str, help='Specify the dataset name')
    parser.add_argument('--run_group_ID', type=str, help='ID of group of runs for W&B')
    parser.add_argument('--training', type=str)
    parser.add_argument('--original_model', type=str)
    parser.add_argument('--model_criterion_name', type=str)
    parser.add_argument('--sae_criterion_name', type=str)
    parser.add_argument('--dead_neurons_steps', type=int)
    parser.add_argument('--sae_checkpoint_epoch', type=int)
    parser.add_argument('--debug_on_cluster', action='store_true', help='Debugging on cluster')
    parser.add_argument('--mis', type=str)
    parser.add_argument('--compute_ie', type=str)
    return parser.parse_args()

if __name__ == '__main__':
    # Parse command line arguments
    args = parse_arguments()

    if args.execution_location is None or args.execution_location == 'local': 
        print("Run code locally")
        use_txt_directly = True 
    elif args.execution_location == 'cluster':
        print("Run code on cluster")
        use_txt_directly = False
    else:
        raise ValueError("Specify a valid execution location.")
    
    if args.debug_on_cluster:
        print("Debugging on cluster")
        use_txt_directly = True


    if use_txt_directly:    
        # create a random (and hopefully unique) group ID
        run_group_ID = "".join(random.choices(string.ascii_lowercase, k=10))

        # Read parameters from the file, line by line, and run the model pipeline
        # consecutively for each parameter combination
        #'''
        with open('parameters.txt', 'r') as file:
            for line in file:
                parameters = [param for param in line.strip().split(',')]
                # make sure that the directory path is a local path
                directory_path = parameters[3]
                if not args.debug_on_cluster and not directory_path.startswith('C:'):
                    raise ValueError(f"The directory path is {parameters[3]} but it should be a local directory path.")
                elif args.debug_on_cluster and not directory_path.startswith('/lustre'):
                    raise ValueError(f"The directory path is {parameters[3]} but it should be a cluster directory path.")
                training = parameters[16]
                original_model = parameters[17]
                #sae_layer = parameters[2]
                #outdated: sae_layer_list = process_sae_layer_list(sae_layer, original_model, training)
                
                # if several SAE layers are specified, we train several SAEs
                #for sae_layer in sae_layer:
                execute_project = ExecuteProject(model_name=parameters[0],
                                                sae_model_name=parameters[1],
                                                sae_layer=parameters[2],
                                                directory_path=directory_path,
                                                wandb_status=parameters[4],
                                                model_epochs=parameters[5],
                                                model_learning_rate=parameters[6],
                                                batch_size=parameters[7],
                                                model_optimizer_name=parameters[8],
                                                sae_epochs=parameters[9],
                                                sae_learning_rate=parameters[10],
                                                sae_optimizer_name=parameters[11],
                                                sae_batch_size=parameters[12],
                                                sae_lambda_sparse=parameters[13],
                                                sae_expansion_factor=parameters[14],
                                                dataset_name=parameters[15],
                                                training=training,
                                                original_model=original_model,
                                                model_criterion_name=parameters[18],
                                                sae_criterion_name=parameters[19],
                                                dead_neurons_steps=parameters[20],
                                                run_group_ID=run_group_ID,
                                                execution_location='local',
                                                mis=parameters[21],
                                                compute_ie=parameters[22],
                                                sae_checkpoint_epoch=parameters[23])
                execute_project.model_pipeline()
        #'''
                
        # Once the model pipeline has been run for all parameter combinations,
        # we can perform evaluation using info from all runs
        '''
        with open('parameters_eval.txt', 'r') as file:
            for line in file:
                parameters_2 = [param for param in line.strip().split(',')]
                # make sure that the directory path is a local path
                directory_path = parameters_2[3]
                if not args.debug_on_cluster and not directory_path.startswith('C:'):
                    raise ValueError(f"The directory path is {directory_path} but it should be a local directory path.")
                elif args.debug_on_cluster and not directory_path.startswith('/lustre'):
                    raise ValueError(f"The directory path is {directory_path} but it should be a cluster directory path.")
                original_model = parameters_2[14]
                #sae_layer=parameters_2[2]
                if not eval(original_model):
                    # Make sure the order of parameters coincides with the one in
                    # parameters_eval.txt (see specify_parameters.py)
                    #for name in sae_layer.split("&"):
                        #if name != "":
                    execute_project = ExecuteProject(model_name=parameters_2[0],
                                                    sae_model_name=parameters_2[1],
                                                    sae_layer=parameters_2[2],
                                                    directory_path=directory_path,
                                                    wandb_status=parameters_2[4],
                                                    model_epochs=parameters_2[5],
                                                    model_learning_rate=parameters_2[6],
                                                    batch_size=parameters_2[7],
                                                    model_optimizer_name=parameters_2[8],
                                                    sae_epochs=parameters_2[9],
                                                    sae_learning_rate=parameters_2[10],
                                                    sae_optimizer_name=parameters_2[11],
                                                    sae_batch_size=parameters_2[12],
                                                    dataset_name=parameters_2[13],
                                                    original_model = original_model,
                                                    run_evaluation=True,
                                                    dead_neurons_steps=parameters_2[15],
                                                    run_group_ID=run_group_ID,
                                                    execution_location='local',
                                                    sae_checkpoint_epoch=parameters_2[16])
                    execute_project.evaluation()
        '''

        # if we used wandb logging, we need to finish the run 
        #if parameters[4]=='True':
        #    wandb.finish()
        
    else:
        #os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb=512" # to avoid CUDA out of memory error

        if args.run_pipeline:
            # make sure that the directory path is a path on the cluster
            if args.directory_path.startswith('C:'): 
                raise ValueError(f"The directory path is {args.directory_path} but it should be a local directory path.")
            elif not args.directory_path.startswith('/lustre'):
                raise ValueError(f"The directory path is {args.directory_path} but it should be a local directory path.")

            # outdated: sae_layer_list = process_sae_layer_list(args.sae_layer, args.original_model, args.training)
            #for name in args.sae_layer:
            execute_project = ExecuteProject(model_name=args.model_name,
                                            sae_model_name=args.sae_model_name,
                                            sae_layer=args.sae_layer,
                                            directory_path=args.directory_path,
                                            wandb_status=args.wandb_status,
                                            model_epochs=args.model_epochs,
                                            model_learning_rate=args.model_learning_rate,
                                            batch_size=args.batch_size,
                                            model_optimizer_name=args.model_optimizer_name,
                                            sae_epochs=args.sae_epochs,
                                            sae_learning_rate=args.sae_learning_rate,
                                            sae_optimizer_name=args.sae_optimizer_name,
                                            sae_batch_size=args.sae_batch_size,
                                            sae_lambda_sparse=args.sae_lambda_sparse,
                                            sae_expansion_factor=args.sae_expansion_factor,
                                            dataset_name=args.dataset_name,
                                            training=args.training,
                                            original_model=args.original_model,
                                            model_criterion_name=args.model_criterion_name,
                                            sae_criterion_name=args.sae_criterion_name,
                                            dead_neurons_steps=args.dead_neurons_steps,
                                            run_group_ID=args.run_group_ID,
                                            execution_location='cluster',
                                            mis=args.mis,
                                            compute_ie=args.compute_ie,
                                            sae_checkpoint_epoch=args.sae_checkpoint_epoch)
            execute_project.model_pipeline()

        #'''
        if args.run_evaluation and not eval(args.original_model):
            # if we only use (or train) the original model, we currently do not perform post-hoc evaluation
            if args.directory_path.startswith('C:'): 
                raise ValueError(f"The directory path is {args.directory_path} but it should be a local directory path.")
            elif not args.directory_path.startswith('/lustre'):
                raise ValueError(f"The directory path is {args.directory_path} but it should be a local directory path.")
            #for name in args.sae_layer.split("&"):
            #    if name != "":
            execute_project = ExecuteProject(model_name=args.model_name,
                                            sae_model_name=args.sae_model_name,
                                            sae_layer=args.sae_layer,
                                            directory_path=args.directory_path,
                                            wandb_status=args.wandb_status,
                                            model_epochs=args.model_epochs,
                                            model_learning_rate=args.model_learning_rate,
                                            batch_size=args.batch_size,
                                            model_optimizer_name=args.model_optimizer_name,
                                            sae_epochs=args.sae_epochs,
                                            sae_learning_rate=args.sae_learning_rate,
                                            sae_optimizer_name=args.sae_optimizer_name,
                                            sae_batch_size=args.sae_batch_size,
                                            dataset_name=args.dataset_name,
                                            original_model=args.original_model,
                                            run_evaluation=args.run_evaluation,
                                            dead_neurons_steps=args.dead_neurons_steps,
                                            run_group_ID=args.run_group_ID,
                                            execution_location='cluster',
                                            sae_checkpoint_epoch=args.sae_checkpoint_epoch)    
            execute_project.evaluation()
        #'''
        if args.wandb_status == 'True':
            wandb.finish()
            print("W&B run finished")