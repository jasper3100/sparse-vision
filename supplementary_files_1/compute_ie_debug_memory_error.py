# not needed code, used at some point for debugging out of memory error, when doing backward twice with nnsight
# for computing ie of edges  
# 
# NOTE: After all, I think the issue just came from me doing .save() on some tensors which I never really needed.
# Then, these tensors accumulated and thus the error.         


    def compute_edge_ie_debug(self, feature_node_threshold=0.0, error_node_threshold=0.0, edge_threshold=0.01):
        # there are not many error terms, so for now we use all of them to get the code right      
        # we will filter nodes based on whether their absolute IE > a node threshold
        sae_features_node_idcs = {}
        sae_error_node_idcs = {}

        # we load the values that were computed before
        encoder_output_average = {}
        sae_error_average = {}
        ie_sae_features = {}
        ie_sae_error = {}
        for name in self.layers:
            encoder_output_average_file_path = get_file_path(self.SAE_encoder_output_averages_folder_path, name, self.params_string[name], '.pt')
            sae_error_average_file_path = get_file_path(self.SAE_error_averages_folder_path, name, self.params_string[name], '.pt')
            ie_sae_features_file_path = get_file_path(self.IE_SAE_features_folder_path, name, self.params_string[name], '.pt')
            ie_sae_error_file_path = get_file_path(self.IE_SAE_errors_folder_path, name, self.params_string[name], '.pt')
 
            encoder_output_average[name] = torch.load(encoder_output_average_file_path) # shape [C*K, H, W]  
            sae_error_average[name] = torch.load(sae_error_average_file_path) # shape [C, H, W]
            ie_sae_features[name] = torch.load(ie_sae_features_file_path) # shape [C*K]
            ie_sae_error[name] = torch.load(ie_sae_error_file_path) # shape: scalar
            
            sae_features_node_idcs[name] = torch.squeeze(torch.nonzero(torch.abs(ie_sae_features[name]) > feature_node_threshold))
            # there is only one SAE error per layer but we just use the same format as for the SAE features
            sae_error_node_idcs[name] = (torch.abs(ie_sae_error[name]) > error_node_threshold).item() # either True or False

        print("-----------------------------")
        print("Nodes left after filtering:")
        for name, tensor in sae_features_node_idcs.items():
            print(name, tensor.numel(), "/", ie_sae_features[name].numel())
        print("-----------------------------")

        batch_idx = 0

        # 4 different types of edges
        ie_feature_d_feature_u = {}
        ie_feature_d_error_u = {}
        ie_error_d_feature_u = {}
        ie_error_d_error_u = {}

        # FOR MAKING THE CODE FASTER I COULD TRY DISABLING GRADIENTS AND ONLY SETTING REQUIRES GRAD TRUE WHERE NEEDED       

        # DEBUG: for debuggin purposes only keep the first index in sae_features_node_idcs
        # except for the first layer
        sae_features_node_idcs = {name: sae_features_node_idcs[name][:2] if name == "mixed3a" else sae_features_node_idcs[name][:1] for name in sae_features_node_idcs}
        print("-----------------------------")
        print("Nodes left after filtering (FOR DEBUGGING):")
        for name, tensor in sae_features_node_idcs.items():
            print(name, tensor.numel(), "/", ie_sae_features[name].numel())
        print("-----------------------------")

        with tqdm(self.dataloader, unit="batch") as tepoch:
            for batch in tepoch:
                print("0", torch.cuda.memory_allocated())
                inputs, targets, _ = process_batch(batch)
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                print("1", torch.cuda.memory_allocated())

                batch_idx += 1

                model_copy = copy.deepcopy(self.model)
                model_copy = model_copy.to(self.device)
                model_copy.eval()
                model_copy = NNsight(model_copy)
                # requires grad for all params in model
                for param in model_copy.parameters():
                    param.requires_grad = True

                grad_original = self.get_grad_original(model_copy, self.layers, inputs, targets, self.debugging, self.model_criterion) # for each layer, the shape is: [N, C, H, W]
                print("2", torch.cuda.memory_allocated())
                print("-----------------------------")

                # we iterate over pairs of consecutive layers
                # we use separate trace contexts for each pair of layers because we don't want a 
                # pass-through gradient on the downstream layer so that we can propagate gradients from SAE to SAE
                for i in range(len(self.layers) - 1):
                    model_copy = copy.deepcopy(self.model)
                    model_copy = model_copy.to(self.device)
                    model_copy.eval()
                    for param in model_copy.parameters():
                        param.requires_grad = False
                    model_copy = NNsight(model_copy)
                    print("9", torch.cuda.memory_allocated())

                    # u = upstream, d = downstream
                    name_u = self.layers[i] 
                    name_d = self.layers[i + 1]

                    # we get the gradient of the loss wrt encoder output of downstream layer
                    # and wrt sae error of downstream layer
                    # we use a separate trace context because in the one below we don't do backprop
                    # from the model loss but from the downstream layer
                    print("3", torch.cuda.memory_allocated())
                    with model_copy.trace(inputs, validate=self.debugging):
                        temp_var_1, encoder_output_d_grad, temp_var_2, sae_error_d_grad = self.intervention(name_d, 
                                                                                        model_copy, 
                                                                                        grad_original, 
                                                                                        save_encoder_output_grad=True,
                                                                                        save_sae_error_grad=True)
                        self.model_criterion(model_copy.output, targets).backward() # backprop so that we have gradients

                    del temp_var_1, temp_var_2
                    
                    # just to be sure we detach the above quantities to treat them as constants
                    print("A", torch.cuda.memory_allocated())
                    encoder_output_d_grad[name_d] = encoder_output_d_grad[name_d].detach() # shape [NHW, C*K]
                    sae_error_d_grad[name_d] = sae_error_d_grad[name_d].detach() # shape [N, C, H, W]
                    print("b", torch.cuda.memory_allocated())

    	            # store the gradients of the different types of edges (between SAE features and SAE errors)
                    prod_grad_feature_d_feature_u = {}
                    prod_grad_feature_d_error_u = {}
                    prod_grad_error_d_feature_u = {}
                    prod_grad_error_d_error_u = {}

                    model_copy = copy.deepcopy(self.model)
                    model_copy = model_copy.to(self.device)
                    model_copy.eval()
                    #for param in model_copy.parameters():
                    #    param.requires_grad = False
                    model_copy = NNsight(model_copy)
                    
                    with model_copy.trace(inputs, validate=self.debugging):
                        # upstream layer
                        # we don't use a stop gradient because we want to access the gradient wrt the sae error
                        encoder_output_u, _, sae_error_u, _ = self.intervention(name_u, 
                                                                                model_copy, 
                                                                                grad_original,
                                                                                save_encoder_output=True,
                                                                                save_sae_error=True,
                                                                                record_sae_error_grad=True)  
                        print("c", torch.cuda.memory_allocated())                  
                        # downstream layer
                        # We don't use a pass-through gradient here because otherwise we wouldn't be able to measure 
                        # the gradient of the upstream encoder output wrt the downstream encoder output
                        encoder_output_d, _, sae_error_d, _ = self.intervention(name_d, 
                                                                                model_copy, 
                                                                                grad_original, 
                                                                                use_pass_through_gradient=False,
                                                                                record_sae_error_grad=True)
                        print("d", torch.cuda.memory_allocated())
                        
                        # encoder_output_d shape: [NHW, C*K]
                        # encoder_output_d_grad (computed in previous trace context) shape: [NHW, C*K]
                        # sae_error_d shape: [NHW, C]
                        # sae_error_d_grad shape: [NHW, C]

                        # edge from d = SAE error in downstream layer to u = SAE feature/SAE error in upstream layer
                        # we can use the same computation graph as above here  
                        # the code is almost the same as above but not exactly
                        '''
                        if sae_error_node_idcs[name_d]:                      
                            d = sae_error_d[name_d] # shape [N, C, H, W]
                            d = rearrange(d, 'b c h w -> (b h w) c 1') # shape [NHW, C, 1]
                            grad_m_d = sae_error_d_grad[name_d] # shape [N, C, H, W]
                            grad_m_d = rearrange(grad_m_d, 'b c h w -> (b h w) 1 c') # shape [NHW, 1, C]

                            # ignoring the NHW dimension, we compute a dot product
                            prod = torch.einsum('nci,nic->n', grad_m_d, d)  # Shape: [NHW]
                            # we take the mean over the batch dimension for doing gradient descent (same as doing backprop
                            # from mean batch loss)
                            prod = torch.mean(prod, dim=0)  # Shape: scalar
                            print(prod)
                            print(prod.requires_grad)

                            # we save the product of the gradient of m wrt d and the gradient of d wrt u 
                            #prod_grad_error_d_feature_u[name_u] = encoder_output_u[name_u].grad.save()
                            #prod_grad_error_d_error_u[name_u] = sae_error_u[name_u].grad.save()

                            # we backprop from the downstream layer (to the upstream layer)
                            prod.backward(retain_graph=True)
                        '''
                        print("-----------------------------")
                        print(torch.cuda.memory_allocated())
                        #'''
                        # edge from d = SAE feature in downstream layer to u = SAE feature/SAE error in upstream layer
                        for sae_feature_idx_d in sae_features_node_idcs[name_d]:
                            sae_feature_idx_d = sae_feature_idx_d.item()

                            d = encoder_output_d[name_d][:, sae_feature_idx_d] # shape [NHW]

                            d = torch.unsqueeze(d, 1) # shape [NHW, 1]
                            grad_m_d = encoder_output_d_grad[name_d][:, sae_feature_idx_d] # shape [NHW]
                            grad_m_d = torch.unsqueeze(grad_m_d, 1) # shape [NHW, 1]

                            d = torch.unsqueeze(d, 2) # shape [NHW, 1, 1]
                            grad_m_d = torch.unsqueeze(grad_m_d, 1) # shape [NHW, 1, 1]

                            # ignoring the NHW dimension, we compute a dot product
                            prod = torch.einsum('nci,nic->n', grad_m_d, d)  # Shape: [NHW]
                            # we take the mean over the batch dimension for doing gradient descent (same as doing backprop
                            # from mean batch loss)
                            prod = torch.mean(prod, dim=0)  # Shape: scalar

                            # we save the product of the gradient of m wrt d and the gradient of d wrt u 
                            #prod_grad_feature_d_feature_u[(name_u, sae_feature_idx_d)] = encoder_output_u[name_u].grad.save() # shape: [NHW, C*K]
                            #prod_grad_feature_d_error_u[(name_u, sae_feature_idx_d)] = sae_error_u[name_u].grad.save() # shape [B, C, H, W]
                        
                            # we backprop from the downstream layer (to the upstream layer)
                            # ATTENTION:  we need to use retain_graph=True when doing backprop several times on the same graph
                            # if sae_feature_idx_d is the last index in the list, we don't need to retain the graph, allowing Pytorch to free memory
                            # otherwise we might get a memory error
                            if sae_feature_idx_d == sae_features_node_idcs[name_d][-1].item():
                                prod.backward()
                                print("Not retaining graph")
                            else: 
                                prod.backward(retain_graph=True)  
                                print("Retaining graph")
                        #self.model_criterion(model_copy.output, targets).backward() # backprop so that we have gradients

                        
                        # We can only do the below if we store the gradient of sae error u and d resp.
                        #sae_error_u[name_u] = sae_error_u[name_u].detach()
                        #sae_error_d[name_d] = sae_error_d[name_d].detach()
                        #'''
                        print(torch.cuda.memory_allocated())
                        
                    #print(sae_error_u)
                    sae_error_u[name_u] = sae_error_u[name_u].detach()
                    #sae_error_d[name_d] = sae_error_d[name_d].detach() # sae_error_d should not BE OUTSIDE HERE, IT#S NOT WHEN I DO BACKPROP THROUGH THE FULL MODEL; BUT IT IS WHEN I DO PARTIAL BACKPROP!!! WHY???

                    # SOMEHOW THE TRACE CONTEXT BEHAVES WEIRDLY WITH PARTIAL BACKPROP!!!

                    del encoder_output_u, sae_error_u
                    del encoder_output_d, sae_error_d # even hough these should not have been saved in the first place???
                    del prod, d, grad_m_d, sae_feature_idx_d, sae_error_d_grad, encoder_output_d_grad
                    print(torch.cuda.memory_allocated())
                    #for param in self.model.parameters():
                    #    param.requires_grad = False
                    #print(torch.cuda.memory_allocated())
                    #for param in self.model.parameters():
                    #    param.requires_grad = True
                    #print(torch.cuda.memory_allocated())
                    del model_copy
                    print(torch.cuda.memory_allocated())
                    print("-----------------------------")



                '''
                    # compute ie from d = SAE feature to u = SAE feature/SAE error

                    TRY THIS!!! with torch.no_grad(): # this is necessary for avoiding out of memory error

                    for sae_feature_idx_d in sae_features_node_idcs[name_d]:
                        sae_feature_idx_d = sae_feature_idx_d.item()
                        # compute ie between two features
                        batch_ie_feature_d_feature_u = compute_ie_channel_wise(encoder_output_u[name_u], # shape [NHW, C*K] 
                                                                                encoder_output_average[name_u], # shape [C*K, H, W] 
                                                                                prod_grad_feature_d_feature_u[(name_u, sae_feature_idx_d)], # shape [NHW, C*K]
                                                                                self.batch_size)
                        # batch_ie_feature_d_feature_u has shape: [C*K], where C is channels in upstream layer
                        # we remove the upstream nodes with low IE 
                        batch_ie_feature_d_feature_u = batch_ie_feature_d_feature_u[sae_features_node_idcs[name_u]]

                        # compute ie between feature (d) and error (u)
                        #batch_ie_feature_d_error_u = compute_ie_all_channels(sae_error_u[name_u], # shape [N, C, H, W]
                        #                                                    sae_error_average[name_u], # shape [C, H, W]
                        #                                                    prod_grad_feature_d_error_u[(name_u, sae_feature_idx_d)],
                        #                                                    self.batch_size) # shape [N, C, H, W]
                        # batch_ie_feature_d_error_u has shape: scalar

                        if name not in ie_feature_d_feature_u:
                            ie_feature_d_feature_u[(name_u,sae_feature_idx_d)] = batch_ie_feature_d_feature_u
                            #ie_feature_d_error_u[(name_u,sae_feature_idx_d)] = batch_ie_feature_d_error_u
                        else: # running average
                            ie_feature_d_feature_u[(name_u,sae_feature_idx_d)] = (ie_feature_d_feature_u[(name_u,sae_feature_idx_d)] * (batch_idx - 1) + batch_ie_feature_d_feature_u) / batch_idx
                            #ie_feature_d_error_u[(name_u,sae_feature_idx_d)] = (ie_feature_d_error_u[(name_u,sae_feature_idx_d)] * (batch_idx - 1) + batch_ie_feature_d_error_u) / batch_idx
                    # compute ie from d = SAE error to u = SAE feature/SAE error
                '''
                '''
                    if sae_error_node_idcs[name_d]:
                        # compute ie between error (d) and feature (u)
                        batch_ie_error_d_feature_u = compute_ie_channel_wise(encoder_output_u[name_u], # shape [NHW, C*K]
                                                                           encoder_output_average[name_u], # shape [C*K, H, W]
                                                                           prod_grad_error_d_feature_u[name_u], # shape [NHW, C*K]
                                                                           self.batch_size)
                        # batch_ie_error_d_feature_u has shape: [C*K], where C is channels in upstream layer
                        # we remove the upstream nodes with low IE 
                        print(encoder_output_u[name_u].shape)
                        print(encoder_output_average[name_u].shape)
                        print(prod_grad_error_d_feature_u[name_u].shape)
                        print(batch_ie_error_d_feature_u.shape)
                        batch_ie_error_d_feature_u = batch_ie_error_d_feature_u[sae_features_node_idcs[name_u]]
                        print(prod_grad_error_d_error_u[name_u].shape)


                        # compute ie between error (d) and error (u)
                        batch_ie_error_d_error_u = compute_ie_all_channels(sae_error_u[name_u], # shape [N, C, H, W]
                                                                          sae_error_average[name_u], # shape [C, H, W]
                                                                          prod_grad_error_d_error_u[name_u],
                                                                          self.batch_size)
                '''
        '''
        for name in self.layers:
            ie_sae_edges_file_path = get_file_path(self.IE_SAE_edges_folder_path, name, self.params_string[name], '.pt')
            # HOW DO WE STORE THE IE OF THESE EDGES???
            #torch.save(ie_sae_features_edges[name], ie_sae_edges_file_path)
            print("Successfully stored IE of SAE feature to SAE feature edges.")
        '''

          



                       
                        