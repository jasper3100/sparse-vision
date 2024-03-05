import os
import numpy as np
import matplotlib.pyplot as plt
import wandb
import torch


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

    bins = np.arange(0, num_classes+1.01, 1)
    print(bins)

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

            axes[row, col].hist(number_active_classes_per_neuron[name].cpu().numpy(), bins=bins, color='blue', edgecolor='black')
            axes[row, col].set_title(f'Layer: {name[0]}, Model: {name[1]}')
            axes[row, col].set_xlabel('Number of active classes')
            axes[row, col].set_ylabel('Number of neurons')

            axes[row, col].set_xticks(np.arange(0.5, num_classes+1, 1))
            axes[row, col].set_xticklabels([str(int(x)) for x in range(num_classes+1)])
    plt.show()

# create some sample data
number_active_classes_per_neuron = {
    ('fc3', 'original'): torch.randint(0, 11, (1000,)),
    ('fc3', 'modified'): torch.randint(0, 10, (1000,)),
    ('fc3', 'sae'): torch.randint(0, 10, (1000,)),
    ('fc2', 'original'): torch.randint(0, 10, (1000,)),
    ('fc2', 'modified'): torch.randint(0, 10, (1000,)),
    ('fc2', 'sae'): torch.randint(0, 10, (1000,))
}

layer_names = ['fc2']
num_classes = 10
plot_active_classes_per_neuron(number_active_classes_per_neuron, layer_names, num_classes)
