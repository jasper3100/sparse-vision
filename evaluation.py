import pandas as pd

from utils import *

class Evaluation:
    '''
    Class to compute evaluation metrics after all runs have been completed,
    i.e., using values from all runs and collecting them.
    '''
    def __init__(self,
                layer_names,
                wandb_status,
                sae_params_1,
                evaluation_results_folder_path):
        self.layer_names = layer_names
        self.wandb_status = wandb_status
        self.sae_params_1 = sae_params_1
        self.evaluation_results_folder_path = evaluation_results_folder_path
        
    def get_sae_eval_results(self, type_of_rec_loss):
        '''
        This function reads SAE eval results (such as nrmse_loss, l1_loss, relative sparsity of SAE encoder output)
        for different lambda_sparse and expansion_factor values from a stored CSV file and visualizes them.
        '''
        # remove lambda_sparse and expansion_factor from params, because we want a uniform file name 
        # for all lambda_sparse and expansion_factor values
        file_path = get_file_path(folder_path=self.evaluation_results_folder_path,
                                    layer_names=self.layer_names,
                                    params=self.sae_params_1,
                                    file_name='sae_eval_results.csv')
        df = pd.read_csv(file_path)

        # if I only want to plot the results for specific expansion factors
        # exlude all rows where expansion factor is 16, 32, or 64
        #df = df[~df['expansion_factor'].isin([16, 32, 64])]

        # adjust the x-axis scale for the plots in column 1 and 3, 
        # if there are more than 3 different expansion factors 
        # (let's see if this is a good rule of thumb, otherwise decide manually)
        def x_scaling_col1(x):
            if len(df['expansion_factor'].unique()) > 3:
                return np.log(x+1) # lambda values, f.e., 0.01, ..., 5 --> log(x+1) expands the interval close to 0
            else:
                return x
        def x_scaling_col3(x):
            if len(df['expansion_factor'].unique()) > 3:
                return x**2 # sparsity values are in [0,1] --> x**2 expands the interval close to 1
            else:   
                return x 

        # Create a subplot grid
        if 'rel_sparsity_1' in df.columns:
            fig, axs = plt.subplots(2, 4, figsize=(21, 10))
        else:
            fig, axs = plt.subplots(2, 3, figsize=(21, 10))

        if type_of_rec_loss == 'mse':
            loss_name = 'rec_loss'
        elif type_of_rec_loss == 'rmse':
            loss_name = 'rmse_loss'
        elif type_of_rec_loss == 'nrmse':
            loss_name = 'nrmse_loss'
        else:
            raise ValueError(f"Invalid type_of_rec_loss: {type_of_rec_loss}")

        ############################### COLUMN 1 ##################################################################
        # Sort the dataframe by lambda_sparse so that the line connects the data points in ascending order
        df_lambda_sparse = df.sort_values(by='lambda_sparse')
        for expansion_factor_value, expansion_factor_group in df_lambda_sparse.groupby('expansion_factor'):
            # Plot rec loss over lambda_sparse with data points
            axs[0, 0].plot(x_scaling_col1(expansion_factor_group['lambda_sparse']), expansion_factor_group[loss_name], label=int(expansion_factor_value))
            axs[0, 0].scatter(x_scaling_col1(expansion_factor_group['lambda_sparse']), expansion_factor_group[loss_name], label='_nolegend_')
            # Plot l1_loss over lambda_sparse with data points
            axs[1, 0].plot(x_scaling_col1(expansion_factor_group['lambda_sparse']), expansion_factor_group['l1_loss'], label=int(expansion_factor_value))
            axs[1, 0].scatter(x_scaling_col1(expansion_factor_group['lambda_sparse']), expansion_factor_group['l1_loss'], label='_nolegend_')

        axs[0, 0].set_xlabel("Lambda Sparse")
        axs[0, 0].set_ylabel(loss_name)
        axs[0, 0].set_title(f"{loss_name} over Lambda Sparse")
        axs[0, 0].legend(title="Expansion Factor", loc='upper left')
        labels = df_lambda_sparse['lambda_sparse'].unique()
        axs[0, 0].set_xticks(x_scaling_col1(labels))
        axs[0, 0].set_xticklabels(labels, rotation='vertical')

        axs[1, 0].set_xlabel("Lambda Sparse")
        axs[1, 0].set_ylabel("L1 Loss")
        axs[1, 0].set_title("L1 Loss over Lambda Sparse")
        axs[1, 0].legend(title="Expansion Factor", loc='upper left')
        axs[1, 0].set_xticks(x_scaling_col1(labels))
        axs[1, 0].set_xticklabels(labels, rotation='vertical')
        
        ############################### COLUMN 2 ##################################################################
        df_expansion_factor = df.sort_values(by='expansion_factor')
        for lambda_sparse_value, lambda_sparse_group in df_expansion_factor.groupby('lambda_sparse'):
            # Plot _loss over expansion_factor with data points
            axs[0, 1].plot(lambda_sparse_group['expansion_factor'], lambda_sparse_group[loss_name], label=lambda_sparse_value)
            axs[0, 1].scatter(lambda_sparse_group['expansion_factor'], lambda_sparse_group[loss_name], label='_nolegend_')
            # Plot l1_loss over expansion_factor with data points
            axs[1, 1].plot(lambda_sparse_group['expansion_factor'], lambda_sparse_group['l1_loss'], label=lambda_sparse_value)
            axs[1, 1].scatter(lambda_sparse_group['expansion_factor'], lambda_sparse_group['l1_loss'], label='_nolegend_')

        axs[0, 1].set_xlabel("Expansion Factor")
        axs[0, 1].set_ylabel(loss_name)
        axs[0, 1].set_title(f"{loss_name} over Expansion Factor")
        axs[0, 1].legend(title="Lambda Sparse", loc='upper left')
        axs[0, 1].set_xticks(df_expansion_factor['expansion_factor'].unique())

        axs[1, 1].set_xlabel("Expansion Factor")
        axs[1, 1].set_ylabel("L1 Loss")
        axs[1, 1].set_title("L1 Loss over Expansion Factor")
        axs[1, 1].legend(title="Lambda Sparse", loc='upper left')
        axs[1, 1].set_xticks(df_expansion_factor['expansion_factor'].unique())

        ############################### COLUMN 3 ##################################################################
        df_rel_sparsity = df.sort_values(by='rel_sparsity')
        # get the min x value
        min_x = df_rel_sparsity['rel_sparsity'].min()
        # round down to the closest fraction of 10
        min_x = np.floor(min_x * 10) / 10
        labels = np.arange(min_x, 1.1, 0.1)
        # for some reason, some labels are 0.30000001 or so, so I round them to 2 decimal places
        labels = [round(label, 2) for label in labels]
        shifted_positions = [x_scaling_col3(label) for label in labels]
            
        for expansion_factor_value, expansion_factor_group in df_rel_sparsity.groupby('expansion_factor'):
            # Plot rec loss over rel_sparsity with data points
            axs[0, 2].plot(x_scaling_col3(expansion_factor_group['rel_sparsity']), expansion_factor_group[loss_name], label=int(expansion_factor_value))
            axs[0, 2].scatter(x_scaling_col3(expansion_factor_group['rel_sparsity']), expansion_factor_group[loss_name], label='__nolegend__')
            # Plot l1_loss over rel_sparsity with data points
            axs[1, 2].plot(x_scaling_col3(expansion_factor_group['rel_sparsity']), expansion_factor_group['l1_loss'], label=int(expansion_factor_value))
            axs[1, 2].scatter(x_scaling_col3(expansion_factor_group['rel_sparsity']), expansion_factor_group['l1_loss'], label='__nolegend__')

        axs[0, 2].set_xlabel(f"Sparsity of SAE encoder output on layer {self.layer_names[0]}")
        axs[0, 2].set_ylabel(loss_name)
        axs[0, 2].set_title(f"{loss_name} over sparsity")
        axs[0, 2].legend(title="Expansion Factor", loc='upper left')
        axs[0, 2].set_xticks(shifted_positions)
        axs[0, 2].set_xticklabels(labels, rotation='vertical')

        axs[1, 2].set_xlabel(f"Sparsity of SAE encoder output on layer {self.layer_names[0]}")
        axs[1, 2].set_ylabel("L1 Loss")
        axs[1, 2].set_title("L1 Loss over sparsity")
        axs[1, 2].legend(title="Expansion Factor", loc='upper left')
        axs[1, 2].set_xticks(shifted_positions)
        axs[1, 2].set_xticklabels(labels, rotation='vertical')

        ############################### COLUMN 4 ##################################################################
        # if there exists a column rel_sparsity_1
        if 'rel_sparsity_1' in df.columns:
            df_rel_sparsity_1 = df.sort_values(by='rel_sparsity_1')
            # get the min x value
            min_x = df_rel_sparsity_1['rel_sparsity_1'].min()
            max_x = df_rel_sparsity_1['rel_sparsity_1'].max()
            # round down to the closest fraction of 10
            min_x = np.floor(min_x * 10) / 10
            max_x = np.ceil(max_x * 10) / 10
            labels = np.linspace(min_x, max_x, 10)
            labels = [round(label, 1) for label in labels]
            shifted_positions = [label for label in labels]
                
            for expansion_factor_value, expansion_factor_group in df_rel_sparsity_1.groupby('expansion_factor'):
                # Plot rec loss over rel_sparsity with data points
                axs[0, 3].plot(expansion_factor_group['rel_sparsity_1'], expansion_factor_group[loss_name], label=int(expansion_factor_value))
                axs[0, 3].scatter(expansion_factor_group['rel_sparsity_1'], expansion_factor_group[loss_name], label='__nolegend__')
                # Plot l1_loss over rel_sparsity with data points
                axs[1, 3].plot(expansion_factor_group['rel_sparsity_1'], expansion_factor_group['l1_loss'], label=int(expansion_factor_value))
                axs[1, 3].scatter(expansion_factor_group['rel_sparsity_1'], expansion_factor_group['l1_loss'], label='__nolegend__')

            axs[0, 3].set_xlabel(f"Sparsity 1 of SAE encoder output on layer {self.layer_names[0]}")
            axs[0, 3].set_ylabel(loss_name)
            axs[0, 3].set_title(f"{loss_name} over sparsity 1")
            axs[0, 3].legend(title="Expansion Factor", loc='upper left')
            axs[0, 3].set_xticks(shifted_positions)
            axs[0, 3].set_xticklabels(labels, rotation='vertical')

            axs[1, 3].set_xlabel(f"Sparsity 1 of SAE encoder output on layer {self.layer_names[0]}")
            axs[1, 3].set_ylabel("L1 Loss")
            axs[1, 3].set_title("L1 Loss over sparsity 1")
            axs[1, 3].legend(title="Expansion Factor", loc='upper left')
            axs[1, 3].set_xticks(shifted_positions)
            axs[1, 3].set_xticklabels(labels, rotation='vertical')        

        # Adjust layout and save the figure
        plt.tight_layout()
        png_path = get_file_path(folder_path=self.evaluation_results_folder_path,
                                layer_names=self.layer_names,
                                params=self.sae_params_1,
                                file_name=f'sae_eval_results_plot_{loss_name}.png')
        plt.savefig(png_path, dpi=300)
        plt.close()

        print(f"Successfully stored SAE eval results plot ({loss_name}) in {png_path}")

        if self.wandb_status:
            # get the keys of self.sae_params_1 and concatenate them into a string with an underscore
            keys = list(self.sae_params_1.keys())
            keys_str = '_'.join(keys)
            wandb.log({f"sae_eval_results_{loss_name}_{keys_str}": wandb.Image(png_path)}, commit=False)