from utils import *
import copy
import sys

# NOTE: I might have confused downstream and upstream at some point in this code. 
# In the forward pass: u --> d (i.e. from upstream to downstream)
# In the backward pass: u <-- d (i.e. from downstream to upstream)

# NOTE: This code was only used for convolutional layers, i.e. with 4 dimensions [N, C, H, W]
# For 2 dimensional model outputs this code might have to be adjusted.

# define a class for computing the IE
class IE:
    def __init__(self, 
                 model, 
                 train_dataloader, 
                 model_params_temp, 
                 device, 
                 sae_model_name, 
                 sae_weights_folder_path, 
                 sae_optimizer_name, 
                 layer_dims_dictionary,
                 ie_related_quantities,
                 model_criterion, 
                 directory_path):
        '''
        1) Compute SAE feature and SAE error average values
        2) Compute the IE for the SAE features and SAE errors
        3) Compute the IE of edges between nodes    
        4) Compute faithfulness of a circuit
        '''
        self.model = model
        self.model.eval()        
        self.model_criterion = model_criterion
        self.device = device
        self.layer_dims_dictionary = layer_dims_dictionary
        print(self.layer_dims_dictionary)
        # create a copy of layer_dims_dictionary 
        #self.layer_dims_dictionary_copy = copy.deepcopy(layer_dims_dictionary)
        # replace each occurence of "mixed" in the keys of the dictionary with "inception"
        #for key in self.layer_dims_dictionary_copy.keys():
        #    self.layer_dims_dictionary_copy[key.replace("mixed", "inception")] = self.layer_dims_dictionary_copy.pop(key)
        #self.layer_dims_dictionary = self.layer_dims_dictionary_copy
        #self.layer_dims_dictionary = {key.replace("mixed", "inception"): value for key, value in self.layer_dims_dictionary.items()}
        #print(self.layer_dims_dictionary)
        self.ie_related_quantities = ie_related_quantities
        self.directory_path = directory_path

        self.dataloader = train_dataloader
        # for computing the ie we use the train dataset. We compute the circuit based on the train dataset and 
        # then validate if it is useful based on the eval dataset. Also, here it doesn't matter if the train dataset is shuffled
        self.layers = ["mixed3a", "mixed3b", "mixed4b", "mixed4c", "mixed4d", "mixed4e", "mixed5a", "mixed5b"] # skip mixed4a for now
        #self.layers = ["inception3a", "inception3b", "inception4a", "inception4b", "inception4c", "inception4d", "inception4e", "inception5a", "inception5b"]
        self.params_string = {}
        self.exp_fac = {}
        
        for folder_name in ["SAE_encoder_output_averages", "SAE_error_averages", "Model_neurons_averages", "IE_SAE_features", "IE_SAE_errors", "IE_Model_neurons", "IE_SAE_edges", "dead_units"]:
            folder_path = os.path.join(ie_related_quantities, folder_name)
            setattr(self, f'{folder_name}_folder_path', folder_path)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

        # we instantiate the SAE models
        for name in self.layers:
            sae_model, self.params_string[name], self.exp_fac[name] = get_specific_sae_model(name,
                                                                                            self.layer_dims_dictionary[name],
                                                                                            sae_model_name,
                                                                                            sae_weights_folder_path,
                                                                                            model_params_temp,
                                                                                            device,
                                                                                            sae_optimizer_name)
            setattr(self, f'{name}_sae', sae_model)

        self.debugging = True # turn off when not debugging to turn off validation argument to be faster

        # SPECIFY THE LAYERS AND FEATURES/NEURONS THAT WE WANT TO CONSIDER (after computing the IE of features; for computing IE edges and for computing the max and min samples) 
        # default (all layers): 
        self.custom_layers = self.layers
        #self.custom_layers = ["mixed3a", "mixed4b", "mixed4c", "mixed5a"] # note that self.layers includes all layers!!!
        # in the other layers we actually also have the sae errors but for simplicity we do not consider them here...
        self.feature_indices = {}
        # default (all features): 
        #for name in self.custom_layers:
        #    self.feature_indices[name] = range(len(ie_sae_features[name]))
        #self.feature_indices["mixed3a"] = [118, 426, 605, 1509]
        #self.feature_indices["mixed4b"] = [948, 1214, 1264, 1287]
        #self.feature_indices["mixed4c"] = [802, 918, 1577, 1847, 1895]
        #self.feature_indices["mixed5a"] = [111, 564, 1054, 1424, 1471, 1606, 1982, 2092, 2569, 2731, 2830]
        # as an alternative to manually specifying the layers and indices we could set the node threshold and compute those values based on the threshold!

        self.compute_top_k_samples = False # CONTINUE HERE BY SETTING TO TRUE AND SEE IF IT WORKS...



    def compute_average(self):
        batch_idx = 0
        num_samples = 0
        encoder_output_average = {}
        sae_error_average = {}
        original_layer_output_average = {}
        sparsity = {}
        dead_units = {}

        model = NNsight(self.model)

        if self.compute_top_k_samples:
            top_k_samples = {}
            filename_txt = os.path.join(self.directory_path, 'dataloaders/imagenet_train_filenames.txt')
            filename_to_idx, self.idx_to_filename = get_string_to_idx_dict(filename_txt)
            # for downloading all flamingo images 
            filename_indices_list = []

        with tqdm(self.dataloader, unit="batch") as tepoch:
            for batch in tepoch:
                inputs, targets, filename_indices = process_batch(batch, 
                                                                  directory_path = self.directory_path)
                                                                  #epoch_mode="compute_max_min_samples", 
                                                                  #filename_to_idx=filename_to_idx) 
                # if the batch is empty, we skip it 
                # f.e. if there are no images of the circuit that we want to consider in the current batch
                if inputs.shape[0] == 0:
                    continue
                else:
                    batch_size = inputs.shape[0]
                    num_samples += batch_size
                inputs, targets, filename_indices = inputs.to(self.device), targets.to(self.device), filename_indices.to(self.device)
                batch_idx += 1
                #if batch_idx > 3:
                #    break

                # GOAL: download all flamingo images into one folder, so that I can use them locally!!!!
                # get all elements out of the tensor filename_indices and add them to an existing list 
                #filename_indices = filename_indices.tolist()
                #filename_indices_list.extend(filename_indices)
                      
                original_layer_output = {}
                with model.trace(inputs, validate=self.debugging):
                    for name in self.layers:
                        original_layer_output[name] = getattr(model, name.replace("mixed", "inception")).output.save()
                        # we only do what is necessary in the trace context to avoid dealing
                        # with issues related to proxy variables etc.
                
                #for name in self.layers:
                #    print(name, original_layer_output[name].shape)

                for name in self.layers:
                    output = original_layer_output[name]
                    b = output.size(0)
                    h = output.size(2)
                    w = output.size(3)
                    output, transformed = reshape_tensor(output)

                    sae = getattr(self, f'{name}_sae')
                    encoder_output, decoder_output, _ = sae(output)
                    sae_error = output - decoder_output
                    batch_dead_units, batch_sparsity, _ = measure_inactive_units(encoder_output, self.exp_fac[name])

                    if transformed: # we need to reshape the quantities to be able to average over the batch dimension        
                        encoder_output = rearrange(encoder_output, '(b h w) c -> b c h w', b=b, h=h, w=w)
                        sae_error = rearrange(sae_error, '(b h w) c -> b c h w', b=b, h=h, w=w)
                    batch_average_sae_error = torch.mean(sae_error, dim=0) # shape: [C, H, W]
                    batch_average_encoder_output = torch.mean(encoder_output, dim=0) # shape [C*K, H, W]
                    batch_average_original_layer_output = torch.mean(original_layer_output[name], dim=0) # shape [C, H, W]

                    '''
                    if self.compute_top_k_samples:    
                        if name in self.custom_layers: # else we do not consider this layer
                            # following model_pipeline.py
                            spatial_average_encoder_output = torch.mean(encoder_output, dim=(2,3)) # shape [B, C*K]
                            # we find the top k samples, there might only be 1 flamingo image in a batch --> batch_size = 1
                            # need to use min(batch_size, 25)
                            batch_topk = torch.topk(spatial_average_encoder_output, k=min(batch_size, 25), dim=0)
                            batch_top_k_values = batch_topk[0]
                            print(name)
                            print(batch_top_k_values)
                            print(batch_top_k_values.shape)
                            batch_top_k_indices = batch_topk[1]
                            print(batch_top_k_indices)
                            print(batch_top_k_indices.shape)
                            print("-----------------")
                            if name not in top_k_samples:
                                top_k_samples[name] = (batch_top_k_values, # dim = 0 -> largest values along the batch dim. (dim=0); 
                                                    batch_top_k_indices,
                                                    batch_size, 
                                                    filename_indices[batch_top_k_indices])
                            else:
                                top_k_samples[name] = get_top_k_samples(top_k_samples[name],
                                                                        batch_top_k_values,
                                                                        batch_top_k_indices,
                                                                        filename_indices[batch_top_k_indices],
                                                                        batch_idx,
                                                                        largest=True,
                                                                        k=25)
                    '''

                    if name not in encoder_output_average:
                        encoder_output_average[name] = batch_average_encoder_output
                        sae_error_average[name] = batch_average_sae_error
                        original_layer_output_average[name] = batch_average_original_layer_output
                        dead_units[name] = batch_dead_units
                        sparsity[name] = batch_sparsity
                    else: # running average
                        encoder_output_average[name] = (encoder_output_average[name] * (num_samples - batch_size) + batch_average_encoder_output * batch_size) / num_samples
                        sae_error_average[name] = (sae_error_average[name] * (num_samples - batch_size) + batch_average_sae_error * batch_size) / num_samples
                        original_layer_output_average[name] = (original_layer_output_average[name] * (num_samples - batch_size) + batch_average_original_layer_output * batch_size) / num_samples
                        dead_units[name] = dead_units[name] * batch_dead_units
                        sparsity[name] = (sparsity[name] * (num_samples - batch_size) + batch_sparsity * batch_size) / num_samples
             
        
        perc_dead_units = {name: torch.sum(dead_units[name]).item() / dead_units[name].shape[0] for name in self.layers}
        path = os.path.join(self.ie_related_quantities, "perc_dead_units_torchvision.csv")
        pd.DataFrame(perc_dead_units.items(), columns=['Layer', 'Percentage dead units']).to_csv(path, index=False)

        path = os.path.join(self.ie_related_quantities, "sparsity_torchvision.csv")
        pd.DataFrame(sparsity.items(), columns=['Layer', 'Sparsity']).to_csv(path, index=False)

        for name in self.layers:
            encoder_output_average_file_path = get_file_path(self.SAE_encoder_output_averages_folder_path, name, self.params_string[name], 'torchvision.pt')
            sae_error_average_file_path = get_file_path(self.SAE_error_averages_folder_path, name, self.params_string[name], 'torchvision.pt')
            original_layer_output_average_file_path = get_file_path(self.Model_neurons_averages_folder_path, name, self.params_string[name], 'torchvision.pt')
            dead_units_file_path = get_file_path(self.dead_units_folder_path, name, self.params_string[name], 'torchvision.pt')
            torch.save(encoder_output_average[name], encoder_output_average_file_path) # shape [C*K, H, W]
            torch.save(sae_error_average[name], sae_error_average_file_path) # shape [C, H, W]
            torch.save(original_layer_output_average[name], original_layer_output_average_file_path) # shape [C, H, W]
            torch.save(dead_units[name], dead_units_file_path)
        print("Successfully stored encoder output averages, SAE error averages and original layer output averages.")

        #unit_idx = 1000
        #name="mixed3a"
        #show_imagenet_images(unit_idx, self.directory_path, self.params_string[name], max_filename_indices=filename_indices_list)  
        '''
        if self.compute_top_k_samples:
            for name in self.custom_layers:
                filename_indices = top_k_samples[name][3]
                # we store the images in a folder
                #for unit_idx in self.feature_indices[name]:

                    show_imagenet_images(unit_idx, self.directory_path, self.params_string[name], max_filename_indices=filename_indices, descending=True)  
        '''             


    def intervention(self, 
                     layer_name, 
                     model, 
                     grad_original=None,
                     use_stop_gradient=True,
                     use_pass_through_gradient=True):      
        # get output of original model layer
        model_output = getattr(model, layer_name.replace("mixed", "inception")).output
        sae = getattr(self, f'{layer_name}_sae')
        encoder_output, decoder_output, _ = apply_sae(sae, model_output)

        # apply stop-gradient on sae error
        sae_error = model_output - decoder_output
        if use_stop_gradient:
            sae_error = sae_error.detach()
            # alternatively can set sae_error.grad[:] = torch.zeros_like(sae_error) (?)
        
        # intervene on model output
        getattr(model, layer_name.replace("mixed", "inception")).output = decoder_output + sae_error
        # or equivalently (?): setattr(getattr(model, layer_name), 'output', decoder_output + sae_error)

        # implement pass-through gradient ([:] is important)
        if use_pass_through_gradient:
            getattr(model, layer_name.replace("mixed", "inception")).output.grad[:] = grad_original[layer_name]

        return encoder_output, sae_error, decoder_output, model_output
    

    def get_grad_original(self, model, layers, inputs, targets, debugging, model_criterion):
        grad_original = {}
        activation_values = {}
        # model without interventions to store the gradients wrt layers of the original model
        with model.trace(inputs, validate=debugging):
            for name in layers:
                # we get the gradients wrt the output of the respective layer
                grad_original[name] = getattr(model, name.replace("mixed", "inception")).output.grad.save()
                activation_values[name] = getattr(model, name.replace("mixed", "inception")).output.save()
                
                #grad_original[name] = getattr(model, name).output.grad.save()
                #activation_values[name] = getattr(model, name).output.save()
                
                # alternatively: setattr(self, f'{name}_grad_original', getattr(model, name.replace("mixed", "inception").output.grad.save())
            #activation_values["softmax2"] = model.softmax2.output.save()
            #grad_original["softmax2"] = model.softmax2.output.grad.save()
            #activation_values["softmax0"] = model.softmax0.output.save()
            #grad_original["softmax0"] = model.softmax0.output.grad.save()
            
            loss = model_criterion(model.output, targets).save()


            #loss_logits = self.model_criterion(model.output.logits, targets)
            #loss_aux_logits1 = self.model_criterion(model.output.aux_logits1, targets)
            #loss_aux_logits2 = self.model_criterion(model.output.aux_logits2, targets)
            #loss = loss_logits + 0.3*loss_aux_logits1 + 0.3*loss_aux_logits2
            #loss.save()
            loss.backward() # backprop so that we have gradients

        #for name in layers:
        #    print(name, torch.median(torch.abs(grad_original[name])), torch.mean(torch.abs(grad_original[name])))
        #print(grad_original["mixed3a"])
        #print("-----------------")
        #print(loss)
        #print("-----------------")
        #for name in layers:
        #    print(torch.median(torch.abs(activation_values[name])), torch.mean(torch.abs(activation_values[name])))
        #name = "softmax2"
        #print(torch.median(activation_values[name]), torch.mean(activation_values[name]))
        #name = "softmax0"
        #print(torch.median(activation_values[name]), torch.mean(activation_values[name]))
        return grad_original

        
    def load(self, return_ie_node_values=True):
        encoder_output_average = {}
        sae_error_average = {}
        original_layer_output_average = {}
        ie_sae_features = {}
        ie_sae_error = {}
        ie_model_neurons = {}
        for name in self.layers:
            encoder_output_average_file_path = get_file_path(self.SAE_encoder_output_averages_folder_path, name, self.params_string[name], 'torchvision.pt')
            sae_error_average_file_path = get_file_path(self.SAE_error_averages_folder_path, name, self.params_string[name], 'torchvision.pt')
            original_layer_output_average_file_path = get_file_path(self.Model_neurons_averages_folder_path, name, self.params_string[name], 'torchvision.pt')
            ie_sae_features_file_path = get_file_path(self.IE_SAE_features_folder_path, name, self.params_string[name], 'torchvision.pt')
            ie_sae_error_file_path = get_file_path(self.IE_SAE_errors_folder_path, name, self.params_string[name], 'torchvision.pt')
            ie_model_neurons_file_path = get_file_path(self.IE_Model_neurons_folder_path, name, self.params_string[name], 'torchvision.pt')
 
            encoder_output_average[name] = torch.load(encoder_output_average_file_path) # shape [C*K, H, W]  
            sae_error_average[name] = torch.load(sae_error_average_file_path) # shape [C, H, W]
            original_layer_output_average[name] = torch.load(original_layer_output_average_file_path) # shape [C, H, W]
            if return_ie_node_values:
                ie_sae_features[name] = torch.load(ie_sae_features_file_path) # shape [C*K]
                ie_sae_error[name] = torch.load(ie_sae_error_file_path) # shape: scalar
                ie_model_neurons[name] = torch.load(ie_model_neurons_file_path)
        
        return encoder_output_average, sae_error_average, ie_sae_features, ie_sae_error, original_layer_output_average, ie_model_neurons
    


    def update_ie_dict(self, ie_vals_dict, name_u, encoder_output_u, encoder_output_average, batch_idx,
                   sae_error_u, sae_error_average,  grad_node_d_feature_u, grad_node_d_error_u, index_d, batch_size):
        # compute ie between downstream node and upstream SAE feature
        batch_ie_node_d_feature_u = compute_ie_channel_wise(encoder_output_u, # shape [NHW, C*K] 
                                                            encoder_output_average, # shape [C*K, H, W] 
                                                            grad_node_d_feature_u, # shape [NHW, C*K]
                                                            batch_size)
        # compute ie between downstream node and upstream SAE error
        batch_ie_node_d_error_u = compute_ie_all_channels(sae_error_u, # shape [N, C, H, W]
                                                        sae_error_average[name_u], # shape [C, H, W]
                                                        grad_node_d_error_u,
                                                        batch_size) # shape [N, C, H, W]
        # batch_ie_error_d_feature_u has shape: [C*K], where C is channels in upstream layer          

        batch_ie_node_d_error_u = batch_ie_node_d_error_u.unsqueeze(0) # convert scalar tensor to a tensor of shape [1]
        batch_ie = torch.cat((batch_ie_node_d_feature_u, batch_ie_node_d_error_u))
        if batch_idx == 1:
            ie_vals_dict[name_u][:, index_d] = batch_ie.save()
        else: # running average
            ie_vals_dict[name_u][:, index_d] = (ie_vals_dict[name_u][:, index_d] * (batch_idx - 1) + batch_ie) / batch_idx
        return ie_vals_dict



    def compute_node_ie(self):
        '''
        As in the reference code by Marks et al. (https://github.com/saprmarks/feature-circuits/blob/cf080789e16f50238097db1fb9a3947a6cfcf9f6/attribution.py#L292) 
        see attribution.py, _pe_attrib function, we apply stop-gradient and pass-through gradient when computing the IE of nodes.
        Without stop-gradient we wouldn't have gradients at the SAE featuers and without pass-through gradients we would, the gradients would
        be distorted quite a bit until they arrive at the desired node.        
        '''
        # we load the encoder output averages and SAE error averages that were computed before (compute_average=True)
        encoder_output_average, sae_error_average, _, _, original_layer_output_average, _ = self.load(return_ie_node_values=False)

        #self.model.train()

        model = NNsight(self.model)
        batch_idx = 0
        num_samples = 0

        ie_sae_features = {}
        ie_sae_error = {}
        ie_model_neurons = {}

        circuit_dead_units = {}
        circuit_dead_units_folder_path = os.path.join(self.ie_related_quantities, "dead_units") # specific to the circuit samples that we use
        for name in self.layers:
            circuit_dead_units_file_path = get_file_path(circuit_dead_units_folder_path, name, self.params_string[name], 'torchvision.pt')
            circuit_dead_units[name] = torch.load(circuit_dead_units_file_path)# map_location=torch.device("cpu")) # shape: [C*K], 


        with tqdm(self.dataloader, unit="batch") as tepoch:
            for batch in tepoch:
                inputs, targets, _ = process_batch(batch, directory_path=self.directory_path)
                # if the batch is empty, we skip it 
                # f.e. if there are no images of the circuit that we want to consider in the current batch
                if inputs.shape[0] == 0:
                    continue
                else:
                    batch_size = inputs.shape[0]
                    num_samples += batch_size
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                # inputs should require grad
                inputs.requires_grad = True

                #print("Median of inputs", torch.median(inputs), torch.mean(inputs))
                #print(inputs)
    
                batch_idx += 1
                #if batch_idx > 3:
                #    break

                #targets = torch.zeros_like(targets)

                # get the gradients of the model loss wrt the model layers (used for pass-through gradients)
                grad_original = self.get_grad_original(model, self.layers, inputs, targets, self.debugging, self.model_criterion)
                               

                for name in self.layers:
                    # for each layer, we do the computation below separately so that we only have to store
                    # the quantities for one layer. Otherwise, it requires too much CUDA memory (unless I store quantities on CPU)
                    with model.trace(inputs, validate=self.debugging):
                        encoder_output, sae_error, _, model_output = self.intervention(name, model, grad_original=grad_original)
                        encoder_output = encoder_output.save()
                        encoder_output_grad = encoder_output.grad.save()
                        model_output = model_output.save()
                        sae_error = sae_error.save()
                                                                                           
                        self.model_criterion(model.output, targets).backward() # backprop so that we have gradients

                    # compute ie and other quantities
                    with torch.no_grad(): # this is necessary for avoiding out of memory error
                        #print(grad_original[name])
                        '''
                        print(encoder_output) # shape [NHW, C*K]
                        print(encoder_output_average[name]) # shape [C*K, H, W]
                        print(encoder_output_grad)
                        print(grad_original[name])
                        print(batch_size)
                        '''
                        batch_ie_sae_features = compute_ie_channel_wise(encoder_output, 
                                                                        encoder_output_average[name], 
                                                                        encoder_output_grad,
                                                                        batch_size)
                        batch_ie_sae_error = compute_ie_all_channels(sae_error, sae_error_average[name], 
                                                                     grad_original[name], batch_size)
                        model_output = rearrange(model_output, 'b c h w -> (b h w) c') 
                        grad_original[name] = rearrange(grad_original[name], 'b c h w -> (b h w) c')  
                        batch_ie_model_neurons = compute_ie_channel_wise(model_output, 
                                                                         original_layer_output_average[name], 
                                                                         grad_original[name], 
                                                                         batch_size)

                        if name not in ie_sae_features:
                            ie_sae_features[name] = batch_ie_sae_features
                            ie_sae_error[name] = batch_ie_sae_error
                            ie_model_neurons[name] = batch_ie_model_neurons
                        else: # running average
                            ie_sae_features[name] = (ie_sae_features[name] * (num_samples - batch_size) + batch_ie_sae_features * batch_size) / num_samples
                            ie_sae_error[name] = (ie_sae_error[name] * (num_samples - batch_size) + batch_ie_sae_error * batch_size) / num_samples
                            ie_model_neurons[name] = (ie_model_neurons[name] * (num_samples - batch_size) + batch_ie_model_neurons * batch_size) / num_samples
                #break
        #'''               
        for name in self.layers:
            ie_sae_features_file_path = get_file_path(self.IE_SAE_features_folder_path, name, self.params_string[name], 'torchvision.pt')
            ie_sae_error_file_path = get_file_path(self.IE_SAE_errors_folder_path, name, self.params_string[name], 'torchvision.pt')
            ie_model_neurons_file_path = get_file_path(self.IE_Model_neurons_folder_path, name, self.params_string[name], 'torchvision.pt')
            torch.save(ie_sae_features[name], ie_sae_features_file_path)
            torch.save(ie_sae_error[name], ie_sae_error_file_path)  
            torch.save(ie_model_neurons[name], ie_model_neurons_file_path)
        print("Successfully stored IE of SAE features, SAE errors and model neurons.")
        #'''         


    def compute_edge_ie(self):
        '''
        As in the reference code by Marks et al. (https://github.com/saprmarks/feature-circuits/blob/cf080789e16f50238097db1fb9a3947a6cfcf9f6/attribution.py#L292) 
        see attribution.py, jvp (jacobian vector product) function, we compute the IE of edges for every pair of layers. As in Marks et al.:
        When computing the gradient of downstream quantity (grad_m_d.T @ d) wrt upstream layer, we don't apply a pass-through or stop-gradient on the downstream 
        layer because we want to measure the gradient of the downstream layer wrt the upstream layer. In the upstream layer, we do apply a stop-gradient after 
        measuring the gradient wrt SAE error so that the gradients also flow through the SAE.        
        '''
        # we load the values that were computed before
        encoder_output_average, sae_error_average, ie_sae_features, ie_sae_error, _, _ = self.load()

        # store edge IE values
        ie_vals_dict = {}

        # we iterate over pairs of consecutive layers
        for i in range(len(self.custom_layers)):
            # u = upstream, d = downstream
            name_u = self.custom_layers[i]
            print(name_u)
            num_sae_features_u = len(self.feature_indices[name_u])
            # we use all features of the upstream layer: expansion factor * number of channels
            #num_sae_features_u = self.exp_fac[name_u] * self.layer_dims_dictionary[name_u]
          
            if name_u != self.custom_layers[-1]:
                name_d = self.custom_layers[i + 1]
                num_sae_features_d = len(self.feature_indices[name_d])
            else: # if i == len(self.custom_layers) - 1, if last layer
                # for the last layer, the downstream layer is the model loss
                # which is not a layer of the model directly (since we pass the ouput to the loss)
                # so we handle this case slightly differently. 
                # Since we have 1 node, for getting the right matrix dimension below, 
                # we should have num_sae_features_d + 1 = 1 --> we set num_sae_features_d = 0
                name_d = "model_loss"
                num_sae_features_d = 0
            
            # the (u_i, d_j) entry of the below matrix refers to edge between u_i and d_j (which is different
            # from the edge between u_j and d_i). The n+1 coordinate is the SAE error
            # shape: [SAE features in u + 1, SAE features in d + 1] = []
            ie_vals_dict[name_u] = torch.zeros(num_sae_features_u + 1, num_sae_features_d + 1)

        batch_idx = 0
        model = NNsight(self.model)


        with tqdm(self.dataloader, unit="batch") as tepoch:
            for batch in tepoch:
                inputs, targets, _ = process_batch(batch, directory_path=self.directory_path)
                # if the batch is empty, we skip it 
                # f.e. if there are no images of the circuit that we want to consider in the current batch
                if inputs.shape[0] == 0:
                    continue
                else:
                    batch_size = inputs.shape[0]
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                batch_idx += 1
                # for debugging
                #if batch_idx == 2:
                #    break

                # we get the gradients of the model loss wrt the model layers (used for pass-through gradients)
                grad_original = self.get_grad_original(model, self.custom_layers, inputs, targets, self.debugging, self.model_criterion) # for each layer, the shape is: [N, C, H, W]

                # we iterate over pairs of consecutive layers
                for i in range(len(self.custom_layers) - 1):
                    
                    # u = upstream, d = downstream
                    name_u = self.custom_layers[i] 
                    name_d = self.custom_layers[i + 1]

                    # we get the gradient of the loss wrt encoder output and sae error of downstream layer
                    # note that gradient wrt sae error = gradient wrt decoder output = gradient wrt layer output = grad_original
                    # we use a separate trace context because in the one below we don't do backprop
                    # from the model loss but from the downstream layer
                    # we don't modify stop- or pass-through gradient here, we compute this gradient as we did it before when 
                    # computing the IE of nodes
                    with model.trace(inputs, validate=self.debugging):
                        encoder_output_d, _, _, _ = self.intervention(name_d, model, grad_original=grad_original)
                        encoder_output_d_grad = encoder_output_d.grad.save()
                        self.model_criterion(model.output, targets).backward() # backprop so that we have gradients
                    # just to be sure we detach to treat them as constants
                    encoder_output_d_grad = encoder_output_d_grad.detach() # shape [NHW, C*K]
                    grad_original[name_d] = grad_original[name_d].detach() # = grad wrt sae error! shape [N, C, H, W]                
                                        
                    # store the prod of grad_m_d*grad_d_u for the different types of edges
                    prod_grad_feature_d_feature_u = {}
                    prod_grad_feature_d_error_u = {}
                    prod_grad_error_d_feature_u = {}
                    prod_grad_error_d_error_u = {}

                    # we do backward pass from downstream to upstream
                    with model.trace(inputs, validate=self.debugging):
                        # upstream layer
                        # We don't use a pass-through gradient here because we don't need it and to possibly avoid any issues
                        # with gradient computations. We don't save any quantities here since we only need them within the trace
                        # context for computing the values we need.
                        encoder_output_u, sae_error_u, decoder_output_u, _ = self.intervention(name_u, model, grad_original=grad_original, 
                                                                                            use_pass_through_gradient=False)                           
                    
                        # downstream layer
                        # We don't use a pass-through gradient here because otherwise we wouldn't be able to measure 
                        # the gradient of d wrt u. Moreover, we don't use a stop-gradient because we also want to measure the gradient
                        # of SAE error wrt upstream node. Not using a stop-gradient is not a problem here, because when doing backward
                        # from an SAE feature, gradients will be computed.
                        encoder_output_d, sae_error_d, _, _ = self.intervention(name_d, model, grad_original=grad_original, 
                                                                            use_stop_gradient=False, use_pass_through_gradient=False)
                        
                        # encoder_output_d shape: [NHW, C*K]
                        # encoder_output_d_grad (grad of loss wrt d, computed in previous trace context) shape: [NHW, C*K]
                        # sae_error_d shape: [NHW, C]

                        # edge from d = SAE feature in downstream layer to u = SAE feature/SAE error in upstream layer

                        for sae_feature_idx_d, i in zip(self.feature_indices[name_d], range(len(self.feature_indices[name_d]))): 
                            d = encoder_output_d[:, sae_feature_idx_d] # shape [NHW]
                            d = torch.unsqueeze(d, 1) # shape [NHW, 1]
                            grad_m_d = encoder_output_d_grad[:, sae_feature_idx_d] # shape [NHW]
                            grad_m_d = torch.unsqueeze(grad_m_d, 1) # shape [NHW, 1]

                            d = torch.unsqueeze(d, 2) # shape [NHW, 1, 1]
                            grad_m_d = torch.unsqueeze(grad_m_d, 1) # shape [NHW, 1, 1]

                            # ignoring the NHW dimension, we compute a dot product
                            prod = torch.einsum('nci,nic->n', grad_m_d, d)  # Shape: [NHW]
                            # we take the mean over the batch dimension for doing gradient descent (same as doing backprop
                            # from mean batch loss)
                            prod = torch.mean(prod, dim=0)  # Shape: scalar

                            # we save the product of the gradient of m wrt d and the gradient of d wrt u 
                            aux_grad = encoder_output_u.grad.save() # shape: [NHW, C*K]
                            # we only keep the entries of the features that we consider in layer u 
                            prod_grad_feature_d_feature_u[(name_u, sae_feature_idx_d)] = aux_grad[:, self.feature_indices[name_u]] # shape: [NHW, len(self.feature_indices[name_u]) <= C*K] 
                            prod_grad_feature_d_error_u[(name_u, sae_feature_idx_d)] = decoder_output_u.grad.save() # shape [B, C, H, W]
                        
                            # we backprop from the downstream layer (to the upstream layer)
                            prod.backward(retain_graph=True)

                            with torch.no_grad():
                                ie_vals_dict = self.update_ie_dict(ie_vals_dict, 
                                                                   name_u, 
                                                                   encoder_output_u[:, self.feature_indices[name_u]], 
                                                                   encoder_output_average[name_u][self.feature_indices[name_u], :, :], 
                                                                   batch_idx, 
                                                                   sae_error_u, 
                                                                   sae_error_average, 
                                                                    grad_node_d_feature_u = prod_grad_feature_d_feature_u[(name_u, sae_feature_idx_d)], # shape [NHW, len(self.feature_indices[name_u])] 
                                                                    grad_node_d_error_u = prod_grad_feature_d_error_u[(name_u, sae_feature_idx_d)], # shape [N, C, H, W]
                                                                    index_d = i,
                                                                    batch_size=batch_size)
                                # encoder_output_u: shape [NHW, C*K]
                                # encoder_output_avergae[name_u]: shape [C*K, H, W]
                                # sae_error_u: shape [N, C, H, W]
                                # sae_error_average[name_u]: shape [C, H, W]

                    
                        # compute ie from d = SAE error to u = SAE feature/SAE error
                        # edge from d = SAE error in downstream layer to u = SAE feature/SAE error in upstream layer
                        # we can use the same computation graph as above here  
                        # the code is almost the same as above but not exactly
                        d = sae_error_d # shape [N, C, H, W]
                        d = rearrange(d, 'b c h w -> (b h w) c 1') # shape [NHW, C, 1]
                        grad_m_d = grad_original[name_d] # = grad of loss wrt sae error! shape [N, C, H, W]
                        grad_m_d = rearrange(grad_m_d, 'b c h w -> (b h w) 1 c') # shape [NHW, 1, C]

                        # ignoring the NHW dimension, we compute a dot product
                        prod = torch.einsum('nci,nic->n', grad_m_d, d)  # Shape: [NHW]
                        # we take the mean over the batch dimension for doing gradient descent (same as doing backprop
                        # from mean batch loss)
                        prod = torch.mean(prod, dim=0)  # Shape: scalar

                        # we save the product of the gradient of m wrt d and the gradient of d wrt u 
                        aux_grad = encoder_output_u.grad.save()
                        prod_grad_error_d_feature_u[name_u] = aux_grad[:, self.feature_indices[name_u]] # shape: [NHW, len(self.feature_indices[name_u]) <= C*K]
                        prod_grad_error_d_error_u[name_u] = decoder_output_u.grad.save()

                        # we backprop from the downstream layer (to the upstream layer)
                        prod.backward() #retain_graph=True)

                        with torch.no_grad():
                            ie_vals_dict = self.update_ie_dict(ie_vals_dict, 
                                                               name_u, 
                                                               encoder_output_u[:, self.feature_indices[name_u]],
                                                               encoder_output_average[name_u][self.feature_indices[name_u], :, :],
                                                               batch_idx, 
                                                               sae_error_u, 
                                                               sae_error_average, 
                                                                grad_node_d_feature_u = prod_grad_error_d_feature_u[name_u], # shape [NHW, len(self.feature_indices[name_u])] 
                                                                grad_node_d_error_u = prod_grad_error_d_error_u[name_u],
                                                                index_d = -1,
                                                                batch_size=batch_size) 

                # if the downstream layer is the model loss, the procedure is simplified
                # since grad_m_d = gradient of loss wrt loss = 1 --> IE: grad_m_u * (u_average - u_activation)
                name_u = self.custom_layers[-1]
                
                # backward pass from model loss to upstream
                # skip this for now because there is an error
                '''
                    ie_vals_dict = self.update_ie_dict(ie_vals_dict, name_u, encoder_output_u, encoder_output_average, batch_idx, sae_error_u, sae_error_average,
                File "/lustre/home/jtoussaint/master_thesis/compute_ie.py", line 269, in update_ie_dict
                    ie_vals_dict[name_u][:, index_d] = batch_ie.save()        
                '''
                #'''
                with model.trace(inputs, validate=self.debugging):
                    # upstream layer (as before)
                    # We don't use a pass-through gradient here because we don't need it and to possibly avoid any issues
                    # with gradient computations. We don't save any quantities here since we only need them within the trace
                    # context for computing the values we need.
                    encoder_output_u, sae_error_u, decoder_output_u, _ = self.intervention(name_u, model, grad_original=grad_original, 
                                                                                        use_pass_through_gradient=False)
                    # gradient from model loss to upstream SAE features
                    aux_grad = encoder_output_u.grad.save() # shape: [NHW, C*K]
                    grad_m_feature_u = aux_grad[:, self.feature_indices[name_u]]
                    # gradient from model loss to upstream SAE error
                    grad_m_error_u = decoder_output_u.grad.save() # shape [B, C, H, W]
                    
                    self.model_criterion(model.output, targets).backward() # backprop so that we have gradients

                    with torch.no_grad():
                        ie_vals_dict = self.update_ie_dict(ie_vals_dict, 
                                                           name_u, 
                                                           encoder_output_u[:, self.feature_indices[name_u]],
                                                           encoder_output_average[name_u][self.feature_indices[name_u], :, :],
                                                           batch_idx, 
                                                           sae_error_u, 
                                                           sae_error_average, 
                                                            grad_node_d_feature_u = grad_m_feature_u, 
                                                            grad_node_d_error_u = grad_m_error_u,
                                                            index_d = 0, # since there is only one downstream node/index  
                                                            batch_size=batch_size)  
                #'''               

        for name in self.custom_layers:
            ie_sae_edges_file_path = get_file_path(self.IE_SAE_edges_folder_path, name, self.params_string[name], '.pt')
            torch.save(ie_vals_dict[name], ie_sae_edges_file_path)
            print("Successfully stored IE of edges.")



    def compute_faithfulness(self, feature_node_threshold=0.0, error_node_threshold=0.0, model_or_sae = "sae"):
        '''
        Given a model M, circuit C, metric (here: model loss) m, we measure the faithfulness of C as
        (m(C) - m(empty)) / (m(M) - m(empty)), where empty = empty circuit.
        We consider 3 different ways of handling the SAE errors: mean ablate, zero ablate, and based on original decoder output.
        '''
        #feature_node_threshold = 0.010880782268941402
        error_node_threshold = feature_node_threshold # I added this only later

        encoder_output_average, sae_error_average, ie_sae_features, ie_sae_error, original_layer_output_average, ie_model_neurons = self.load()

        # filtering nodes
        sae_features_node_idcs = {}
        sae_error_node_idcs = {}
        model_neurons_node_idcs = {}
        for name in self.layers:
            sae_features_node_idcs[name] = torch.abs(ie_sae_features[name]) > feature_node_threshold # tensor of True/False
            sae_error_node_idcs[name] = (torch.abs(ie_sae_error[name]) > error_node_threshold).item() # value: True/False
            model_neurons_node_idcs[name] = torch.abs(ie_model_neurons[name]) > feature_node_threshold # tensor of True/False
        
        print("-----------------------------")
        if model_or_sae == "sae":
            num_sae_error_nodes = sum(sae_error_node_idcs.values())
            print("Number of SAE error nodes included in circuit:", num_sae_error_nodes, "/", len(sae_error_node_idcs))
            print("-----------------------------")
            print("Nodes left after filtering:")
            for name, tensor in sae_features_node_idcs.items():
                print(name, torch.sum(tensor).item(), "/", len(tensor))
        elif model_or_sae == "model":
            for name, tensor in model_neurons_node_idcs.items():
                print(name, torch.sum(tensor).item(), "/", len(tensor))
        print("-----------------------------")


        # DEBUG: for debuggin purposes only keep the first 5 True element in sae_features_node_idcs (for each layer)
        '''
        for name in sae_features_node_idcs:
            tensor = sae_features_node_idcs[name]
            # Find the indices of the True elements
            true_indices = torch.where(tensor)[0]

            # Create a mask with all False
            mask = torch.zeros_like(tensor, dtype=bool)

            # Set the first 5 True elements in the mask
            if len(true_indices) > 0:
                mask[true_indices[:5]] = True

            # Apply the mask to the original tensor
            result_tensor = tensor & mask
            sae_features_node_idcs[name] = result_tensor
         
        print("-----------------------------")
        print("Nodes left after filtering (FOR DEBUGGING):")
        for name, tensor in sae_features_node_idcs.items():
            print(name, torch.sum(tensor).item(), "/", len(tensor))
        print("-----------------------------")
        '''

        batch_idx = 0
        nnsight_model = NNsight(self.model)

        # we don't need any gradients here as we only do forward passes
        # without torch.no_grad we get memory errors (likely since we store gradients somewhere)
        with torch.no_grad():

            with tqdm(self.dataloader, unit="batch") as tepoch:
                for batch in tepoch:
                    inputs, targets, _ = process_batch(batch, directory_path=self.directory_path)
                    # if the batch is empty, we skip it 
                    # f.e. if there are no images of the circuit that we want to consider in the current batch
                    if inputs.shape[0] == 0:
                        continue
                    inputs, targets = inputs.to(self.device), targets.to(self.device)

                    batch_idx += 1
                    #if batch_idx > 3: 
                    #    break

                    if model_or_sae == "sae":
                        #print("Zero ablation")
                        # C with all SAE errors zero ablated (i.e. without SAE errors)
                        with nnsight_model.trace(inputs, validate=self.debugging):
                            for name in self.layers: 
                                model_output = getattr(nnsight_model, name.replace("mixed", "inception")).output
                                sae = getattr(self, f'{name}_sae')
                                # we set those nodes (SAE features/encoder output) which are below the IE threshold to the average value
                                _, decoder_output, new_decoder_output = apply_sae(sae, 
                                                                        model_output,
                                                                        nodes=sae_features_node_idcs[name], 
                                                                        ablation=encoder_output_average[name])
                                # intervene on model output
                                #decoder_output_saved = decoder_output.save()
                                #new_decoder_output_saved = new_decoder_output.save()
                                getattr(nnsight_model, name.replace("mixed", "inception")).output = new_decoder_output # no sae error
                            m_C_sae_errors_zero_ablated = self.model_criterion(nnsight_model.output, targets).item().save()
                        
                        # check if decoder_output and new_decoder_output are the same 
                        #print(torch.allclose(decoder_output_saved, new_decoder_output_saved, atol=1e-8))

                        
                        #print("Mean ablation")
                        # C with all SAE errors mean ablated
                        with nnsight_model.trace(inputs, validate=self.debugging):
                            for name in self.layers: 
                                model_output = getattr(nnsight_model, name.replace("mixed", "inception")).output
                                sae = getattr(self, f'{name}_sae')
                                _, _, new_decoder_output = apply_sae(sae, 
                                                                        model_output,
                                                                        nodes=sae_features_node_idcs[name], 
                                                                        ablation=encoder_output_average[name])
                                # intervene on model output
                                getattr(nnsight_model, name.replace("mixed", "inception")).output = new_decoder_output + sae_error_average[name]
                            m_C_sae_errors_mean_ablated = self.model_criterion(nnsight_model.output, targets).item().save()
                        
                        # C 
                        # How we handle SAE errors: We compute SAE error given the original decoder output, and for those SAE errors which are below 
                        # IE threshold we set them to the average value. We don't compute the SAE error based on ablated decoder output because 
                        # then the model output would not change, and thus measuring m(C) would not make sense. 
                        with nnsight_model.trace(inputs, validate=self.debugging):
                            for name in self.layers: 
                                model_output = getattr(nnsight_model, name.replace("mixed", "inception")).output
                                sae = getattr(self, f'{name}_sae')
                                _, decoder_output, new_decoder_output = apply_sae(sae,
                                                                                    model_output, 
                                                                                    nodes=sae_features_node_idcs[name], 
                                                                                    ablation=encoder_output_average[name])
                                # compute SAE error based on original decoder output
                                sae_error = model_output - decoder_output
                                # mean-ablate SAE errors
                                if not sae_error_node_idcs[name]: # if IE below threshold
                                    sae_error = sae_error_average[name]
                                # intervene on model output
                                getattr(nnsight_model, name.replace("mixed", "inception")).output = new_decoder_output + sae_error
                            m_C = self.model_criterion(nnsight_model.output, targets).item().save()
                    
                        # empty circuit, i.e., all SAE features and SAE errors are set to the average value
                        with nnsight_model.trace(inputs, validate=self.debugging):
                            for name in self.layers: 
                                model_output = getattr(nnsight_model, name.replace("mixed", "inception")).output
                                sae = getattr(self, f'{name}_sae')
                                _, decoder_output, new_decoder_output = apply_sae(sae,
                                                                                    model_output,
                                                                                    nodes=torch.zeros_like(sae_features_node_idcs[name], dtype=torch.bool), 
                                                                                    ablation=encoder_output_average[name])
                                # intervene on model output
                                getattr(nnsight_model, name.replace("mixed", "inception")).output = new_decoder_output + sae_error_average[name]
                            m_empty = self.model_criterion(nnsight_model.output, targets).item().save()
                    
                    elif model_or_sae == "model":
                        # C 
                        with nnsight_model.trace(inputs, validate=self.debugging):
                            for name in self.layers: 
                                model_output = getattr(nnsight_model, name.replace("mixed", "inception")).output
                                # mean ablate model neurons below threshold
                                # model_output has shape [B, C, H, W], original_layer_output_average[name] has shape [C, H, W], model_neurons_node_idcs[name] has shape [C]
                                model_output[:, ~model_neurons_node_idcs[name], : , :] = original_layer_output_average[name][~model_neurons_node_idcs[name], :, :]
                                # intervene on model output
                                getattr(nnsight_model, name.replace("mixed", "inception")).output = model_output
                            m_C = self.model_criterion(nnsight_model.output, targets).item().save()
                        
                        # empty model 
                        with nnsight_model.trace(inputs, validate=self.debugging):
                            for name in self.layers:
                                model_output = getattr(nnsight_model, name.replace("mixed", "inception")).output
                                model_output[:, :, :, :] = original_layer_output_average[name][:, :, :]
                                # intervene on model output
                                getattr(nnsight_model, name.replace("mixed", "inception")).output = model_output
                            m_empty = self.model_criterion(nnsight_model.output, targets).item().save()

                    # full model
                    m_M = self.model_criterion(self.model(inputs), targets).item()

                    # keep running averages of all losses
                    if batch_idx == 1:
                        if model_or_sae == "sae":
                            m_C_sae_errors_zero_ablated_avg = m_C_sae_errors_zero_ablated
                            m_C_sae_errors_mean_ablated_avg = m_C_sae_errors_mean_ablated
                        m_C_avg = m_C
                        m_empty_avg = m_empty
                        m_M_avg = m_M
                    else:
                        if model_or_sae == "sae":
                            m_C_sae_errors_zero_ablated_avg = (m_C_sae_errors_zero_ablated_avg * (batch_idx - 1) + m_C_sae_errors_zero_ablated) / batch_idx
                            m_C_sae_errors_mean_ablated_avg = (m_C_sae_errors_mean_ablated_avg * (batch_idx - 1) + m_C_sae_errors_mean_ablated) / batch_idx
                        m_C_avg = (m_C_avg * (batch_idx - 1) + m_C) / batch_idx
                        m_empty_avg = (m_empty_avg * (batch_idx - 1) + m_empty) / batch_idx
                        m_M_avg = (m_M_avg * (batch_idx - 1) + m_M) / batch_idx

            # compute faithfulness of the different circuits
            # faithfulness = (m(C) - m(empty)) / (m(M) - m(empty))
            if model_or_sae == "sae":
                faithfulness_sae_errors_zero_ablated = (m_C_sae_errors_zero_ablated_avg - m_empty_avg) / (m_M_avg - m_empty_avg)
                faithfulness_sae_errors_mean_ablated = (m_C_sae_errors_mean_ablated_avg - m_empty_avg) / (m_M_avg - m_empty_avg)
            faithfulness = (m_C_avg - m_empty_avg) / (m_M_avg - m_empty_avg)   

            print("Threshold values:", feature_node_threshold, error_node_threshold)
            if model_or_sae == "sae":
                print("Faithfulness (SAE errors zero ablated):", faithfulness_sae_errors_zero_ablated)
                print("Faithfulness (SAE errors mean ablated):", faithfulness_sae_errors_mean_ablated)
            print("Faithfulness:", faithfulness)

            # store faithfulness values, and the threshold values in an excel file
            if model_or_sae == "sae":
                file_path = os.path.join(self.ie_related_quantities, "faithfulness.xlsx")
            else: 
                file_path = os.path.join(self.ie_related_quantities, "faithfulness_model.xlsx")
            if not os.path.exists(file_path):
                if model_or_sae == "sae":
                    df = pd.DataFrame(columns=["feature_node_threshold", "error_node_threshold", "faithfulness_sae_errors_zero_ablated", "faithfulness_sae_errors_mean_ablated", "faithfulness"])
                    df.loc[0] = [feature_node_threshold, error_node_threshold, faithfulness_sae_errors_zero_ablated, faithfulness_sae_errors_mean_ablated, faithfulness]
                else: 
                    df = pd.DataFrame(columns=["feature_node_threshold", "error_node_threshold", "faithfulness"])
                    df.loc[0] = [feature_node_threshold, error_node_threshold, faithfulness]
            else:
                df = pd.read_excel(file_path) 
                # check if the threshold values are already in the file
                if not ((df["feature_node_threshold"] == feature_node_threshold) & (df["error_node_threshold"] == error_node_threshold)).any():
                    if model_or_sae == "sae":
                        df.loc[len(df)] = [feature_node_threshold, error_node_threshold, faithfulness_sae_errors_zero_ablated, faithfulness_sae_errors_mean_ablated, faithfulness]
                    else:
                        df.loc[len(df)] = [feature_node_threshold, error_node_threshold, faithfulness]
                else: # overwrite the faithfulness values
                    idx = df[(df["feature_node_threshold"] == feature_node_threshold) & (df["error_node_threshold"] == error_node_threshold)].index[0]
                    if model_or_sae == "sae":
                        df.loc[idx, "faithfulness_sae_errors_zero_ablated"] = faithfulness_sae_errors_zero_ablated
                        df.loc[idx, "faithfulness_sae_errors_mean_ablated"] = faithfulness_sae_errors_mean_ablated
                    df.loc[idx, "faithfulness"] = faithfulness
            df.to_excel(file_path, index=False)
            print(f"Successfully stored faithfulness values in {file_path}.")
    
    def plot_faithfulness(self):
        file_path = os.path.join(self.ie_related_quantities, "faithfulness.xlsx")
        df = pd.read_excel(file_path)
        df = df.sort_values(by=["feature_node_threshold", "error_node_threshold"])
        df = df.reset_index(drop=True)

        fig, ax = plt.subplots(1, 3, figsize=(20, 5))
        for i, col in enumerate(["faithfulness_sae_errors_zero_ablated", "faithfulness_sae_errors_mean_ablated", "faithfulness"]):
            ax[i].plot(df["feature_node_threshold"], df[col], label=col)
            ax[i].set_xlabel("Feature node threshold")
            ax[i].set_ylabel("Faithfulness")
            ax[i].set_title(col)
            ax[i].legend()
        plt.tight_layout()
        plt.show()

    def visualize_circuit(self):
        # check out the plot_circuit.py script by Marks et al...
        pass