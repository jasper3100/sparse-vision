
def plot_active_classes_per_neuron(number_active_classes_per_neuron, 
                                   layer_names, 
                                   num_classes,
                                   folder_path=None, 
                                   params=None, 
                                   wandb_status=None):
    # for now, only show results for the specified layer and the last layer, which is 'fc3'
    # 2 rows for the 2 layers we're considering, and 3 columns for the 3 types of models (original, modified, sae)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Number of active classes per neuron')

    number_bins = 20
    bin_width = num_classes / number_bins
    # Set bin edges, we start from a small constant > 0 because we create a separate bin
    # for zero values (dead neurons) to distinguish between dead neurons and ultra sparse neurons
    # For example, if num_classes = 10 --> bin_width = 0.5 --> bin edges: [epsilon, 0.5), [0.5, 1.0), ...
    bins1 = np.array([np.finfo(float).eps, bin_width])
    bins2 = np.arange(bin_width, num_classes+0.0001, bin_width) 
    # adding some small value to end point because otherwise np.arange does not include the end point
    bins = np.concatenate((bins1, bins2))

    for name in number_active_classes_per_neuron.keys():
        if name[0] in layer_names or name[0] == 'fc3':
            if name[0] == 'fc3':
                row = 1
            else:
                row = 0
            if name[1] == 'original':
                col = 0
            elif name[1] == 'modified':
                col = 1
            elif name[1] == 'sae':
                col = 2

            axes[row, col].hist(number_active_classes_per_neuron[name].cpu().numpy(), bins=bins, color='blue')
            axes[row, col].set_title(f'Layer: {name[0]}, Model: {name[1]}')
            axes[row, col].set_xlabel('Number of active classes')
            axes[row, col].set_ylabel('Number of neurons')
            # set a custom tick label of the tick at x=np.finfo(float).eps
            x_ticks = axes[row, col].get_xticks()
            axes[row, col].set_xticks(np.append(x_ticks, (0, -0.5))) # add a tick at x=0 and x=-0.5
            axes[row, col].set_xticklabels([str(int(x)) for x in x_ticks] + [r' $\epsilon$', '0']) # label at x=0 is epsilon and at x=-0.5 is 0

            # Add a red bar for values equal to zero (dead neurons)
            dead_neurons = (number_active_classes_per_neuron[name].cpu().numpy() == 0).sum()
            axes[row, col].bar(-0.25, dead_neurons, color='red', width=0.5, label=f'{dead_neurons} dead neurons')
            axes[row, col].legend(loc='upper right')

    plt.subplots_adjust(wspace=0.7)  # Adjust space between images
    
    if wandb_status:
        wandb.log({"active_classes_per_neuron":wandb.Image(plt)})
    else:
        os.makedirs(folder_path, exist_ok=True)
        file_path = get_file_path(folder_path, layer_names, params, 'active_classes_per_neuron.png')
        plt.savefig(file_path)
        plt.close()
        print(f"Successfully stored active classes per neuron plot in {file_path}")
