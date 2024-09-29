import pandas as pd
import matplotlib.lines as mlines
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator

from utils import *

class Evaluation:
    '''
    Class to compute evaluation metrics after all runs have been completed,
    i.e., using values from all runs and collecting them.
    '''
    def __init__(self,
                sae_layer,
                wandb_status,
                model_params,
                sae_params_1,
                evaluation_results_folder_path,
                sae_checkpoint_epoch):
        self.sae_layer = sae_layer
        self.wandb_status = wandb_status
        self.model_params = model_params
        self.sae_params_1 = sae_params_1
        self.evaluation_results_folder_path = evaluation_results_folder_path
        self.sae_checkpoint_epoch = sae_checkpoint_epoch
        model_params = {k: str(v) for k, v in self.model_params.items()}
        sae_params_1 = {k: str(v) for k, v in self.sae_params_1.items()}
        self.params_string_1 = '_'.join(model_params.values()) + "_" + "_".join(sae_params_1.values())
        self.file_path = get_file_path(folder_path=self.evaluation_results_folder_path,
                                    sae_layer=self.sae_layer,
                                    params=self.params_string_1,
                                    file_name='sae_eval_results.csv')
        #print("Opening file: ", self.file_path)
    
    def compute_sae_ranking(self):
        '''
        This function reads SAE eval results for different lambda, exp factor, learning rate,... 
        from a stored CSV file and computes a ranking of the different SAEs
        '''       
        df = pd.read_csv(self.file_path)

        # compute the ranks for each metric
        # ascending=False: highest value gets the best/highest rank (=1)
        # ascending=True: lowest value gets the best/highest rank (=1)
        df['var_expl_rank'] = df["var_expl"].rank(ascending=False).astype(int)
        df['l1_rank'] = df['l1_loss'].rank(ascending=True).astype(int)
        df['rec_loss_rank'] = df['nrmse_loss'].rank(ascending=True).astype(int) # for the rec loss we take the nrmse loss
        df['perc_dead_units_rank'] = df['perc_dead_units'].rank(ascending=True).astype(int)
        df['sparsity_rank'] = df['rel_sparsity'].rank(ascending=False).astype(int)
        df['loss_diff_rank'] = df['loss_diff'].rank(ascending=True).astype(int)
        df['mis_rank'] = df['median_mis'].rank(ascending=False).astype(int) 

        df['average_ranking'] = df[['var_expl_rank', 'l1_rank', 'rec_loss_rank', 'perc_dead_units_rank', 'sparsity_rank', 'loss_diff_rank', 'mis_rank']].mean(axis=1)
        df['final_ranking'] = df['average_ranking'].rank(ascending=True).astype(int)
        df = df.sort_values(by='final_ranking')

        rank_table_path = get_file_path(folder_path=self.evaluation_results_folder_path,
                                        sae_layer=self.sae_layer,
                                        params=self.params_string_1,
                                        file_name=f'sae_rank_table.csv')
        df.to_csv(rank_table_path, index=False)
        if self.wandb_status:
            wandb.log({f"eval/sae_eval_results/{self.params_string_1}": wandb.Table(dataframe=df)}, commit=False)
        print(f"Successfully computed and stored SAE ranking in {rank_table_path}")

        
    def plot_rec_loss_vs_sparsity(self, type_of_rec_loss): #, type_of_sparsity):
        '''
        This function reads SAE eval results (such as nrmse_loss, l1_loss, sparsity of SAE encoder output)
        for different lambda_sparse and expansion_factor values from a stored CSV file and visualizes them.
        '''
        df = pd.read_csv(self.file_path)

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
        fig, axs = plt.subplots(2, 4, figsize=(21, 10))
        fig.suptitle(f"SAE Evaluation Results for Layer {self.sae_layer} at Epoch {self.sae_checkpoint_epoch}")

        if type_of_rec_loss == 'mse':
            loss_name = 'rec_loss'
        elif type_of_rec_loss == 'rmse':
            loss_name = 'rmse_loss'
        elif type_of_rec_loss == 'nrmse':
            loss_name = 'nrmse_loss'
        else:
            raise ValueError(f"Invalid type_of_rec_loss: {type_of_rec_loss}")

        # get the unique expansion factors and assign a color to each of them
        exp_fac_colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']
        exp_fac_to_color = dict(zip(sorted(df['expansion_factor'].unique()), exp_fac_colors)) # f.e. {2.0: 'blue', 4.0: 'orange', ...}
        # prepare legend with expansion factors
        exp_fac_legend_list = []
        for items in exp_fac_to_color.items():
            exp_fac = items[0]
            color = items[1]
            exp_fac_legend_list.append(mlines.Line2D([], [], color=color, marker='o', linestyle='None', label=f'{exp_fac}'))
        # get the unique lambda_sparse values and assign a color to each of them
        lambda_colors = ['olive', 'cyan', 'magenta', 'yellow', 'black', 'pink']
        lambda_to_color = dict(zip(sorted(df['lambda_sparse'].unique()), lambda_colors)) # f.e. {0.01: 'gray', 0.02: 'olive', ...}
        # prepare legend with lambda_sparse values
        lambda_legend_list = []
        for items in lambda_to_color.items():
            lambda_value = items[0]
            color = items[1]
            lambda_legend_list.append(mlines.Line2D([], [], color=color, marker='o', linestyle='None', label=f'{lambda_value}'))
        # add a column to the dataframe that contains the colors for each row
        df['exp_fac_color'] = df['expansion_factor'].map(exp_fac_to_color)
        df['lambda_color'] = df['lambda_sparse'].map(lambda_to_color)     

        # only keep those rows of df where the column "epochs" has the value self.sae_checkpoint_epoch
        df = df[df['epochs'] == float(self.sae_checkpoint_epoch)]
        #print(df)
        #print("-----------------")


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
        max_x = df_rel_sparsity['rel_sparsity'].max()
        max_x = np.ceil(max_x * 10) / 10

        # for better comparability between plots of one layer but several epochs
        # we set axis limits
        min_x, max_x, min_y, max_y = self.layer_specific_axes_limits()
        axs[0, 2].set_ylim(min_y, max_y)
        axs[0, 2].set_xlim(min_x, max_x)

        labels = np.linspace(min_x, max_x, 5)
        # for some reason, some labels are 0.30000001 or so, so I round them to 2 decimal places
        labels = [round(label, 2) for label in labels]
        shifted_positions = [x_scaling_col3(label) for label in labels]      
        
        # choose whatever looks better/more understandable in the plot
        # parameter connected by a line
        param_1 = "expansion_factor"
        label_1 = "Exp fac"
        color_1 = "exp_fac_color"
        # parameter with colored dots
        label_2 = "Lambda"
        color_2 = "lambda_color" # should be the param of the dots
        legend_2 = lambda_legend_list

        for value, group in df_rel_sparsity.groupby(param_1):
            # Plot rec loss over rel_sparsity with data points
            line_color = group[color_1].iloc[0] # group[color_1] is a series with the same color for all rows, but here we should pass one color value
            axs[0, 2].plot(x_scaling_col3(group['rel_sparsity']), group[loss_name], label=int(value), color=line_color)
            axs[0, 2].scatter(x_scaling_col3(group['rel_sparsity']), group[loss_name], color=group[color_2], label='__nolegend__')
            # Plot l1_loss over rel_sparsity with data points
            axs[1, 2].plot(x_scaling_col3(group['rel_sparsity']), group['l1_loss'], label=int(value))
            axs[1, 2].scatter(x_scaling_col3(group['rel_sparsity']), group['l1_loss'], label='__nolegend__')

        axs[0, 2].set_xlabel(f"Sparsity of SAE encoder output on layer {self.sae_layer}")
        axs[0, 2].set_ylabel(loss_name)
        axs[0, 2].set_title(f"{loss_name} over sparsity")
        axs[0, 2].legend(title=label_1, loc='upper right')
        axs[0, 2].set_xticks(shifted_positions)
        axs[0, 2].set_xticklabels(labels, rotation='vertical')

        # insert a second, separate legend
        ax2 = axs[0, 2].twinx()
        ax2.legend(handles = legend_2, title=label_2, loc='center right')
        ax2.axis('off')

        axs[1, 2].set_xlabel(f"Sparsity of SAE encoder output on layer {self.sae_layer}")
        axs[1, 2].set_ylabel("L1 Loss")
        axs[1, 2].set_title("L1 Loss over sparsity")
        axs[1, 2].legend(title="Lambda", loc='upper left')
        axs[1, 2].set_xticks(shifted_positions)
        axs[1, 2].set_xticklabels(labels, rotation='vertical')

        ############################### COLUMN 4 ##################################################################
        df_l1_loss = df.sort_values(by='l1_loss')
        # get the min x value
        min_x = df_l1_loss['l1_loss'].min()
        max_x = df_l1_loss['l1_loss'].max()
        # round down to the closest fraction of 10
        min_x = np.floor(min_x * 10) / 10
        max_x = np.ceil(max_x * 10) / 10
        labels = np.linspace(min_x, max_x, 5)
        labels = [round(label, 1) for label in labels]
        shifted_positions = [label for label in labels]
            
        for expansion_factor_value, expansion_factor_group in df_l1_loss.groupby('expansion_factor'):
            # Plot rec loss over l1 loss with data points
            line_color = expansion_factor_group['exp_fac_color'].iloc[0] # group[color_1] is a series with the same color for all rows, but here we should pass one color value
            axs[0, 3].plot(expansion_factor_group['l1_loss'], expansion_factor_group[loss_name], label=int(expansion_factor_value), color=line_color)
            axs[0, 3].scatter(expansion_factor_group['l1_loss'], expansion_factor_group[loss_name], color=expansion_factor_group['lambda_color'], label='__nolegend__')

        axs[0, 3].set_xlabel(f"L1 loss of SAE encoder output on layer {self.sae_layer}")
        axs[0, 3].set_ylabel(loss_name)
        axs[0, 3].set_title(f"{loss_name} over L1 loss")
        axs[0, 3].legend(title="Expansion Factor", loc='upper left')
        axs[0, 3].set_xticks(shifted_positions)
        axs[0, 3].set_xticklabels(labels, rotation='vertical') 

        # insert a second, separate legend
        ax2 = axs[0, 3].twinx()
        ax2.legend(handles = lambda_legend_list, title="Lambda", loc='upper right')

        #################################################################################################     

        # Adjust layout and save the figure
        plt.tight_layout()
        png_path = get_file_path(folder_path=self.evaluation_results_folder_path,
                                sae_layer=self.sae_layer,
                                params=self.params_string_1,
                                file_name=f'sae_eval_results_plot_{loss_name}_epoch_{self.sae_checkpoint_epoch}.png')
        if self.wandb_status:
            wandb.log({f"eval/sae_eval_results/{self.params_string_1}/{loss_name}/epoch_{self.sae_checkpoint_epoch}": wandb.Image(plt)}, commit=False)
        plt.show()
        #plt.savefig(png_path, dpi=300)
        plt.close()
        print(f"Successfully stored SAE eval results plot ({loss_name}) of epoch {self.sae_checkpoint_epoch} in {png_path}")



    def plot_rec_loss_vs_sparsity_all_epochs(self, type_of_rec_loss):
        df = pd.read_csv(self.file_path)

        # set universal font size
        plt.rcParams.update({'font.size': 12})

        if "mixed3a" in self.file_path:
            epochs = [7,15,20,25,30,35] 
            file_ending = "many_epochs_v1"
            #epochs = [37,40,42,45,48,50] 
            #file_ending = "many_epochs_v2"
        elif "mixed3b" in self.file_path:
            epochs = [15, 20, 25, 28, 30]
            file_ending = "many_epochs_v1"
            #epochs = [32, 35, 37, 40, 42, 45] 
            #file_ending = "many_epochs_v2"
            #epochs = [47, 50, 52, 55, 57, 60]
            #file_ending = "many_epochs_v3"

        # default choice. All image processing and pixel related choices 
        # are tuned for this choice.
        rows = 2
        cols = 3

        fig = plt.figure(figsize=(14,7))
        gs = GridSpec(rows, cols, figure=fig, wspace=0, hspace=0)

        if type_of_rec_loss == 'mse':
            loss_name = 'rec_loss'
        elif type_of_rec_loss == 'rmse':
            loss_name = 'rmse_loss'
        elif type_of_rec_loss == 'nrmse':
            loss_name = 'nrmse_loss'
        else:
            raise ValueError(f"Invalid type_of_rec_loss: {type_of_rec_loss}")

        # get the unique expansion factors and assign a color to each of them
        exp_fac_colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']
        exp_fac_to_color = dict(zip(sorted(df['expansion_factor'].unique()), exp_fac_colors))
        # prepare legend with expansion factors
        exp_fac_legend_list = []
        for items in exp_fac_to_color.items():
            exp_fac = items[0]
            color = items[1]
            exp_fac_legend_list.append(mlines.Line2D([], [], color=color, marker='o', linestyle='None', label=f'{exp_fac}'))
        # get the unique lambda_sparse values and assign a color to each of them
        lambda_colors = ['olive', 'cyan', 'magenta', 'yellow', 'black', 'pink']
        lambda_to_color = dict(zip(sorted(df['lambda_sparse'].unique()), lambda_colors))
        # prepare legend with lambda_sparse values
        lambda_legend_list = []
        for items in lambda_to_color.items():
            lambda_value = items[0]
            color = items[1]
            lambda_legend_list.append(mlines.Line2D([], [], color=color, marker='o', linestyle='None', label=f'{lambda_value}'))
        # add a column to the dataframe that contains the colors for each row
        df['exp_fac_color'] = df['expansion_factor'].map(exp_fac_to_color)
        df['lambda_color'] = df['lambda_sparse'].map(lambda_to_color)     

        counter = 0
        axs = []

        # keep only the rows where the column "epochs" has a value in the list epochs
        df = df[df['epochs'].isin(epochs)]

        # get the min x value
        min_x = df['rel_sparsity'].min()
        # round down to the closest fraction of 10
        min_x = np.floor(min_x * 10) / 10
        max_x = df['rel_sparsity'].max()
        max_x = np.ceil(max_x * 10) / 10

        min_y = df['nrmse_loss'].min()
        min_y = np.floor(min_y * 1000) / 1000
        max_y = df['nrmse_loss'].max()
        max_y = np.ceil(max_y * 1000) / 1000

        # if the axis labels don't look "nice", f.e., value at boundary, we can adjust them
        y_labels = np.linspace(min_y + (max_y - min_y) / 10, max_y - (max_y - min_y) / 10, 4)
        y_labels = [round(label, 3) for label in y_labels]
        use_custom_y_ticks = True
        x_labels = np.linspace(min_x + (max_x - min_x) / 10, max_x - (max_x - min_x) / 10, 3)
        x_labels = [round(label, 2) for label in x_labels]
        use_custom_x_ticks = True

        for epoch in epochs:
            # create a copy of the dataframe
            df_copy = df.copy()
            # only keep the rows where the column "epochs" has the value epoch
            df_copy = df_copy[df_copy['epochs'] == float(epoch)]

            df_rel_sparsity = df_copy.sort_values(by='rel_sparsity')

            # for better comparability between plots of one layer but several epochs
            # we set axis limits
            #min_x, max_x, min_y, max_y = self.layer_specific_axes_limits()
    
            # choose whatever looks better/more understandable in the plot
            # parameter connected by a line
            param_1 = "expansion_factor"
            label_1 = "Exp fac"
            color_1 = "exp_fac_color"
            # parameter with colored dots
            label_2 = "Lambda"
            color_2 = "lambda_color" # should be the param of the dots
            legend_2 = lambda_legend_list

            row_idx = counter // cols
            col_idx = counter % cols

            ax = fig.add_subplot(gs[row_idx, col_idx])
            axs.append(ax)

            for value, group in df_rel_sparsity.groupby(param_1):
                # Plot rec loss over rel_sparsity with data points
                line_color = group[color_1].iloc[0]
                #ax.plot(x_scaling_col3(group['rel_sparsity']), group[loss_name], label=int(value), color=line_color)
                ax.plot(group['rel_sparsity'], group[loss_name], label=int(value), color=line_color)
                # ax.scatter(x_scaling_col3(group['rel_sparsity']), group[loss_name], color=group[color_2], label='__nolegend__')
                ax.scatter(group['rel_sparsity'], group[loss_name], color=group[color_2], label='__nolegend__')
            
            ax.set_ylim(min_y, max_y)
            ax.set_xlim(min_x, max_x)
            # Rotate x-axis tick labels for better readability
            ax.tick_params(axis='x', rotation=45)
            # Reduce number of y-axis ticks
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4))

            # add some custom text in the top middle
            ax.text(0.5, 0.93, f"Epoch {epoch}", ha="center", transform=ax.transAxes)

            if use_custom_y_ticks:
                ax.set_yticks(y_labels)
                ax.set_yticklabels(y_labels)
            if use_custom_x_ticks:
                ax.set_xticks(x_labels)
                ax.set_xticklabels(x_labels, rotation=45)

                
            if row_idx == 0 and col_idx == cols - 1:
                ''' # might overlap!
                # add legend
                ax.legend(title=label_1, loc='lower right', labelspacing=0.4) #handles = exp_fac_legend_list
                # insert a second, separate legend
                ax2 = ax.twinx()
                # location should be lower right
                ax2.legend(handles = legend_2, title=label_2, loc='upper right', labelspacing=0.4)
                # this axis should not have ticks
                ax2.axis('off')
                '''
                
                legend1 = ax.legend(title=label_1, loc='upper right', bbox_to_anchor=(1, 0.4), framealpha=1) # set transparency to 1 so that the background does not shine through
                legend2 = ax.legend(handles= legend_2, title=label_2, loc='upper right', bbox_to_anchor=(1, 1), framealpha=1)
                # Add legends to the figure
                fig.add_artist(legend1)
                fig.add_artist(legend2)



            # Hide x-ticks for all but the last row
            if row_idx != rows - 1:
                ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

            # Hide y-ticks for all but the first column
            if col_idx != 0:
                ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)

            counter += 1

        
        fig.text(0.5, 0.02, "Density", ha="center") #, color="white") # sparsity of SAE encoder output
        if loss_name == 'nrmse_loss':
            fig.text(0.06, 0.5, "Rec. loss (NRMSE)", va="center", rotation="vertical")
        else:
            fig.text(0.06, 0.5, f"Rec. loss ({loss_name})", va="center", rotation="vertical")

        # Adjust layout and save the figure
        plt.subplots_adjust(wspace=0, hspace=0)
        #plt.show()
        png_path = get_file_path(folder_path=self.evaluation_results_folder_path,
                                 sae_layer=self.sae_layer,
                                 params=self.params_string_1,
                                 file_name=f'sae_eval_results_plot_{loss_name}_{file_ending}.png')
        plt.savefig(png_path, dpi=300)
        plt.close()
        print(f"Successfully stored SAE eval results plot ({loss_name}) of epoch {self.sae_checkpoint_epoch} in {png_path}")


    def layer_specific_axes_limits(self):
        # for better comparability between plots of one layer but several epochs
        # we set axis limits
        if "mixed3a" in self.file_path:
            if self.sae_checkpoint_epoch >= 37:
                min_x = 0.65
                max_x = 1.1
                max_y = 0.035
                min_y = 0.0 
            elif self.sae_checkpoint_epoch > 7:
                min_y = 0.0
                max_y = 0.055
                min_x = 0.5
                max_x = 1.7
        if "mixed3b" in self.file_path and self.sae_checkpoint_epoch >= 37:
            min_x = 0.77
            max_x = 0.92
            max_y = 0.022
            min_y = 0.008
        elif "mixed4a" in self.file_path and self.sae_checkpoint_epoch >= 37:
            max_y = 0.023
            min_y = 0.012
            min_x = 0.75
            max_x = 0.9 
        elif "mixed4b" in self.file_path and self.sae_checkpoint_epoch >= 35:
            min_x = 0.7
            max_x = 0.9
            min_y = 0.014
            max_y = 0.026
        elif "mixed4c" in self.file_path and self.sae_checkpoint_epoch >= 35:
            min_x = 0.7
            max_x = 0.95
            min_y = 0.011
            max_y = 0.025
        
        return min_x, max_x, min_y, max_y