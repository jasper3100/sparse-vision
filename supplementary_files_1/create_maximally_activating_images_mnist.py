

# this only worked for MNIST but it would have to be adapted now because self.only_get_act is not defined anymore as I'm not storing
# the activations in the model_pipeline class anymore as this is too expensive for larger datasets
    def create_maximally_activating_images(self, 
                                            layer_name, 
                                            model_key, 
                                            top_k_samples, # used for getting dead neurons
                                            folder_path,
                                            layer_names,
                                            params,
                                            number_neurons,
                                            wandb_status):
        # We define this function in this class because otherwise we would have to pass the model as input
        # register the hooks, take care to only return the activations etc.

        # The few lines below are copied from the show_top_k_samples function
        top_value_matrix = top_k_samples[(layer_name, model_key)][0]
        # we store the column indices of non-dead neurons --> we will not plot activating samples for dead neurons because this is pointless
        non_zero_columns = torch.any(top_value_matrix != 0, dim=0) # --> tensor([True, False, True,...]), True wherever non-zero
        # the number of neurons we can plot is upper bounded by the number of non-dead neurons in the layer
        number_non_dead_neurons = torch.sum(non_zero_columns).item()
        number_neurons = min(number_neurons, number_non_dead_neurons)
        # find the column indices of the first number_neurons non-zero columns
        non_zero_columns_indices = torch.nonzero(non_zero_columns).squeeze()[:number_neurons]

        # so far this function only works for MNIST!!!
        MNIST_mean = 0.1307
        MNIST_std = 0.3081

        num_cols, num_rows = rows_cols(number_neurons)
        fig = plt.figure(figsize=(18,9))
        outer_grid = fig.add_gridspec(num_rows, num_cols, wspace=0.1, hspace=0.3)
        fig.suptitle(f'Maximally activating image for {number_neurons} neurons \n in {model_key, layer_name}', fontsize=10)
        
        # for specifying in the loop that we only want to get the activations
        self.only_get_act = True

        for i in range(num_rows):
            for j in range(num_cols):
                if i * num_cols + j >= number_neurons:
                    break
                neuron_idx = non_zero_columns_indices[i * num_cols + j].item()

                #outer_grid_1 = outer_grid[i,j].subgridspec(1, 3, wspace=0, hspace=0, width_ratios=[1, 0.1, 1])
                outer_grid_1 = outer_grid[i,j].subgridspec(1, 1, wspace=0, hspace=0)#width_ratios=[1, 0.1, 1])

                # add ghost axis --> allows us to add a title
                ax_title_1 = fig.add_subplot(outer_grid_1[:])
                ax_title_1.set(xticks=[], yticks=[])
                ax_title_1.set_title(f'Neuron {neuron_idx}\n', fontsize=9, pad=0) # we add a newline so that title is above the ones of the subplots

                # we generate the maximally and minimally activating image for the current neuron
                for mode in ['max']:#,'min']:
                    if mode == "max":
                        inner_grid = outer_grid_1[0, 0].subgridspec(1, 2, wspace=0, hspace=0)#, height_ratios=[1,1], width_ratios=[1,1])
                    #else:
                    #    inner_grid = outer_grid_1[0, 2].subgridspec(1, 2, wspace=0, hspace=0)#, height_ratios=[1,1], width_ratios=[1,1])

                    # add ghost axis --> allows us to add a title
                    ax_title_2 = fig.add_subplot(inner_grid[:])
                    ax_title_2.set(xticks=[], yticks=[])
                    ax_title_2.set_title(f'{mode} activating', fontsize=8, pad=0.1)

                    axs = inner_grid.subplots()  # Create all subplots for the inner grid.
                    version = 0
                    for (c), ax in np.ndenumerate(axs): # if there are only rows/columns use (c) instead of (c,d)
                        ax.set(xticks=[], yticks=[])
                        ax.set_box_aspect(1) # make the image square
                        ax.axis('off')
                        version += 1
                        if version > 2:
                            break
                        optim_image = torch.randn(28*28) 
                        optim_image.data = processed_optim_image(optim_image.data, str(version), MNIST_mean, MNIST_std, True)
                        #optim_image = optim_image.view(1, 28*28)
                        optim_image.requires_grad = True
                        
                        # we define an optimizer for optimizing the image
                        optimizer = torch.optim.SGD([optim_image], lr=0.1)
                        
                        # Get the maximally activating image
                        for iteration in range(500):
                            if model_key=='original' and self.use_sae:
                                self.model_copy(optim_image)
                            else:
                                self.model(optim_image)
                            act = self.batch_activations[(layer_name,model_key)]
                            #print(self.batch_activations)
                            #print(act.shape)
                            # we use the pre-relu encoder output, because otherwise we would have many zeros and we couldn't 
                            # optimize the image effectively

                            # remove singleton dimensions: [1,neuron_idx] --> [neuron_idx] (1 was for the one sample we consider)
                            act = act.squeeze()
                            # consider the output of the neuron we're interested in
                            act = act[neuron_idx]
                            if mode == 'max':
                                # since we want to maximize the activation value of this neuron, we minimize the negative of the activation value
                                loss = -act
                            elif mode == 'min':
                                loss = act
                            else:
                                raise ValueError("Invalid mode")
                            
                            #print(act, layer_name, model_key)

                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()

                            optim_image.data = processed_optim_image(optim_image.data, str(version), MNIST_mean, MNIST_std, False)

                        optim_image = optim_image.detach().numpy().reshape(28, 28)                   
                        ax.imshow(optim_image, cmap='gray', aspect='auto')  # Assuming grayscale images
        #'''
        if wandb_status:
            wandb.log({f"eval/generated_max_and_min_activating_images/{model_key}_{layer_name}":wandb.Image(plt)})
        else:
            folder_path = os.path.join(folder_path, 'generated_max_and_min_activating_images')
            os.makedirs(folder_path, exist_ok=True)
            file_path = get_file_path(folder_path, layer_names, params, f'{model_key}_{layer_name}_{number_neurons}_generated_max_min_activating_images.png')
            plt.savefig(file_path, bbox_inches='tight', pad_inches=0.1, dpi=300)
            print(f"Successfully stored generated max and min activating image for {number_neurons} neurons in {file_path}")
        plt.close()
        #'''
        self.only_get_act = False # to be sure we just turn this parameter off again