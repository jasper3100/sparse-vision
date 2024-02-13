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
        
    def get_sae_eval_results(self):
        '''
        This function reads SAE eval results (such as rec_loss, l1_loss, relative sparsity of SAE encoder output)
        for different lambda_sparse and expansion_factor values from a stored CSV file and visualizes them.
        '''
        # remove lambda_sparse and expansion_factor from params, because we want a uniform file name 
        # for all lambda_sparse and expansion_factor values
        file_path = get_file_path(folder_path=self.evaluation_results_folder_path,
                                    layer_names=self.layer_names,
                                    params=self.sae_params_1,
                                    file_name='sae_eval_results.csv')
        df = pd.read_csv(file_path)

        # Create a 2x3 subplot grid
        fig, axs = plt.subplots(2, 3, figsize=(18, 10))

        # Sort the dataframe by lambda_sparse so that the line connects the data points in ascending order
        df_lambda_sparse = df.sort_values(by='lambda_sparse')
        for expansion_factor_value, expansion_factor_group in df_lambda_sparse.groupby('expansion_factor'):
            # Plot rec_loss over lambda_sparse with data points
            axs[0, 0].plot(expansion_factor_group['lambda_sparse'], expansion_factor_group['rec_loss'], label=f"Factor {expansion_factor_value}")
            axs[0, 0].scatter(expansion_factor_group['lambda_sparse'], expansion_factor_group['rec_loss'], label='_nolegend_')
            # Plot l1_loss over lambda_sparse with data points
            axs[1, 0].plot(expansion_factor_group['lambda_sparse'], expansion_factor_group['l1_loss'], label=f"Factor {expansion_factor_value}")
            axs[1, 0].scatter(expansion_factor_group['lambda_sparse'], expansion_factor_group['l1_loss'], label='_nolegend_')

        axs[0, 0].set_xlabel("Lambda Sparse")
        axs[0, 0].set_ylabel("Rec Loss")
        axs[0, 0].set_title("Rec Loss over Lambda Sparse")
        axs[0, 0].legend(title="Expansion Factor", labels=df_lambda_sparse['expansion_factor'].unique(), loc='upper left')
        axs[0, 0].set_xticks(df_lambda_sparse['lambda_sparse'].unique())

        axs[1, 0].set_xlabel("Lambda Sparse")
        axs[1, 0].set_ylabel("L1 Loss")
        axs[1, 0].set_title("L1 Loss over Lambda Sparse")
        axs[1, 0].legend(title="Expansion Factor", labels=df_lambda_sparse['expansion_factor'].unique(), loc='upper left')
        axs[1, 0].set_xticks(df_lambda_sparse['lambda_sparse'].unique())
        
        df_expansion_factor = df.sort_values(by='expansion_factor')
        for lambda_sparse_value, lambda_sparse_group in df_expansion_factor.groupby('lambda_sparse'):
            # Plot rec_loss over expansion_factor with data points
            axs[0, 1].plot(lambda_sparse_group['expansion_factor'], lambda_sparse_group['rec_loss'], label=f"Lambda Sparse {lambda_sparse_value}")
            axs[0, 1].scatter(lambda_sparse_group['expansion_factor'], lambda_sparse_group['rec_loss'], label='_nolegend_')
            # Plot l1_loss over expansion_factor with data points
            axs[1, 1].plot(lambda_sparse_group['expansion_factor'], lambda_sparse_group['l1_loss'], label=f"Lambda Sparse {lambda_sparse_value}")
            axs[1, 1].scatter(lambda_sparse_group['expansion_factor'], lambda_sparse_group['l1_loss'], label='_nolegend_')

        axs[0, 1].set_xlabel("Expansion Factor")
        axs[0, 1].set_ylabel("Rec Loss")
        axs[0, 1].set_title("Rec Loss over Expansion Factor")
        axs[0, 1].legend(title="Lambda Sparse", labels=df_expansion_factor['lambda_sparse'].unique(), loc='upper left')
        axs[0, 1].set_xticks(df_expansion_factor['expansion_factor'].unique())

        axs[1, 1].set_xlabel("Expansion Factor")
        axs[1, 1].set_ylabel("L1 Loss")
        axs[1, 1].set_title("L1 Loss over Expansion Factor")
        axs[1, 1].legend(title="Lambda Sparse", labels=df_expansion_factor['lambda_sparse'].unique(), loc='upper left')
        axs[1, 1].set_xticks(df_expansion_factor['expansion_factor'].unique())

        #df_rel_sparsity = df.sort_values(by='rel_sparsity')
        # Plot rec_loss over rel_sparsity
        axs[0, 2].plot(df['rel_sparsity'], df['rec_loss'])# label='_nolegend_')
        axs[0, 2].scatter(df['rel_sparsity'], df['rec_loss'])#, label='_nolegend_')
        # Plot l1_loss over rel_sparsity
        axs[1, 2].plot(df['rel_sparsity'], df['l1_loss'])# label='_nolegend_')
        axs[1, 2].scatter(df['rel_sparsity'], df['l1_loss'])#, label='_nolegend_')

        axs[0, 2].set_xlabel(f"Relative sparsity of SAE encoder output on layer {self.layer_names[0]}")
        axs[0, 2].set_ylabel("Rec Loss")
        axs[0, 2].set_title("Rec Loss over relative sparsity")
        axs[0, 2].set_xticks(df['rel_sparsity'].unique())

        axs[1, 2].set_xlabel(f"Relative sparsity of SAE encoder output on layer {self.layer_names[0]}")
        axs[1, 2].set_ylabel("L1 Loss")
        axs[1, 2].set_title("L1 Loss over relative sparsity")
        axs[1, 2].set_xticks(df['rel_sparsity'].unique())

        # Adjust layout and save the figure
        plt.tight_layout()
        png_path = get_file_path(folder_path=self.evaluation_results_folder_path,
                                layer_names=self.layer_names,
                                params=self.sae_params_1,
                                file_name='sae_eval_results_plot.png')
        plt.savefig(png_path, dpi=300)
        plt.close()

        print(f"Successfully stored SAE eval results plot in {png_path}")

        if self.wandb_status:
            wandb.log({f"sae_eval_results_{self.sae_params_1}": wandb.Image(png_path)}, commit=False)