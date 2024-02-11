import argparse
import random
import string
import wandb

from execute_project import ExecuteProject

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
    parser.add_argument('--use_sae', type=str)
    parser.add_argument('--train_sae', type=str)
    parser.add_argument('--train_original_model', type=str)
    parser.add_argument('--store_activations', type=str)
    parser.add_argument('--compute_feature_similarity', type=str)
    parser.add_argument('--model_criterion_name', type=str)
    parser.add_argument('--sae_criterion_name', type=str)
    # if we are running the code on the cluster we can specify whether to run pipeline and/or evaluation
    # MAKE THIS PARAMETER A BOOLEAN FLAG!!!
    parser.add_argument('--run_pipeline', type=str)
    parser.add_argument('--run_evaluation', type=str)
    return parser.parse_args()

if __name__ == '__main__':
    # Parse command line arguments
    args = parse_arguments()

    if args.execution_location is None or args.execution_location == 'local': 
        print("Run code locally")
        # create a random (and hopefully unique) group ID
        run_group_ID = "".join(random.choices(string.ascii_lowercase, k=10))

        # Read parameters from the file, line by line, and run the model pipeline
        # consecutively for each parameter combination
        with open('parameters.txt', 'r') as file:
            for line in file:
                parameters = [param for param in line.strip().split(',')]
                #execute_project = ExecuteProject(*parameters, run_group_ID)
                execute_project = ExecuteProject(model_name=parameters[0],
                                                sae_model_name=parameters[1],
                                                layer_names=parameters[2],
                                                directory_path=parameters[3],
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
                                                activation_threshold=parameters[15],
                                                dataset_name=parameters[16],
                                                use_sae=parameters[17],
                                                train_sae=parameters[18],
                                                train_original_model=parameters[19],
                                                store_activations=parameters[20],
                                                compute_feature_similarity=parameters[21],
                                                model_criterion_name=parameters[22],
                                                sae_criterion_name=parameters[23],
                                                run_group_ID=run_group_ID)
                execute_project.model_pipeline()

        # Once the model pipeline has been run for all parameter combinations,
        # we can perform evaluation using info from all runs
        with open('parameters_eval.txt', 'r') as file:
            for line in file:
                parameters_2 = [param for param in line.strip().split(',')]
                use_sae = parameters_2[-1]
                if eval(use_sae):
                    # Make sure the order of parameters coincides with the one in
                    # parameters_eval.txt (see specify_parameters.py)
                    execute_project = ExecuteProject(model_name=parameters_2[0],
                                                    sae_model_name=parameters_2[1],
                                                    layer_names=parameters_2[2],
                                                    directory_path=parameters_2[3],
                                                    wandb_status=parameters_2[4],
                                                    sae_epochs=parameters_2[5],
                                                    sae_learning_rate=parameters_2[6],
                                                    sae_optimizer_name=parameters_2[7],
                                                    sae_batch_size=parameters_2[8],
                                                    activation_threshold=parameters_2[9],
                                                    dataset_name=parameters_2[10],
                                                    run_group_ID=run_group_ID)
                    execute_project.evaluation()

        if parameters[4]=='True' or parameters_2[4]=='True':
            wandb.finish()
        
    elif args.execution_location == 'cluster':
        print("Run code on cluster")
        if args.run_pipeline:
            execute_project = ExecuteProject(args.model_name,
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
                                            args.use_sae,
                                            args.train_sae,
                                            args.train_original_model,
                                            args.store_activations,
                                            args.compute_feature_similarity,
                                            args.model_criterion_name,
                                            args.sae_criterion_name,
                                            args.run_group_ID)
            execute_project.model_pipeline()
        if args.run_evaluation:
            execute_project = ExecuteProject(args.model_name,
                                            args.sae_model_name,
                                            args.layer_names,
                                            args.directory_path,
                                            args.wandb_status,
                                            args.sae_epochs,
                                            args.sae_learning_rate,
                                            args.sae_optimizer_name,
                                            args.sae_batch_size,
                                            args.activation_threshold,
                                            args.dataset_name,
                                            args.run_group_ID)
            execute_project.evaluation()
        if args.wandb_status == 'True':
            wandb.finish()
    else:
        raise ValueError("Please specify a valid execution location: either 'local' or 'cluster' or None")