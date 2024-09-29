# INSTANTIATE SAE MODELS ALLOWING FOR SEQUENTIALLY TRAINING THEM
'''
Examples of "sae layers" values
'fc1&fc2&&' # original model with SAEs on fc1 and fc2
'&&fc3' # take original model and train SAE on fc3
'fc1&fc2&&fc3' # take original model with SAEs on fc1 and fc2, and train SAE on fc3
'fc1&fc2&&fc3&fc4' # take original model with SAEs on fc1 and fc2, and first train SAE on fc3, then take original model with SAEs on fc1, fc2, fc3 and train SAE on fc4
if we use the original model, sae_layers will be reset to some default value
'''
# IN THE CURRENT CODE THIS IS NOT POSSIBLE ANYMORE (only possible to train one SAE at a time, not sequentially)
# but using several SAEs at once is possible...

        if self.use_sae:
            if self.training:
                # split self.layer_names on the last occurence of "&"
                if "&" in self.layer_names:
                    pretrained_sae_layers_string, self.train_sae_layer = self.layer_names.rsplit('&', 1)
                else:
                    pretrained_sae_layers_string = ""
                    self.train_sae_layer = self.layer_names
                if pretrained_sae_layers_string == "":
                    print("Training SAE on layer", self.train_sae_layer, "and not using any pretrained SAEs.")
                else:
                    print("Training SAE on layer", self.train_sae_layer, "and using pretrained SAEs on layers", pretrained_sae_layers_string)
            else:
                if self.layer_names.endswith("&&"):
                    pretrained_sae_layers_string = self.layer_names[:-2]
            
            # Split the string into a list based on '&'
            self.pretrained_sae_layers_list = pretrained_sae_layers_string.split("&")

            # since the SAE models are trained sequentially, we need to load the pretrained SAEs in the correct order
            # f.e. if pretrained_sae_layers_list = ["layer1", "layer2", "layer3", "layer4"], then we need to load the 
            # SAEs with names: "layer1", "layer1&layer2", "layer1&layer2&layer3", "layer1&layer2&layer3&layer4"
            self.pretrained_saes_list = []
            for i in range(len(self.pretrained_sae_layers_list)):
                joined = '&'.join(self.pretrained_sae_layers_list[:i+1])
                self.pretrained_saes_list.append(joined)

            self.sae_criterion = get_criterion(sae_criterion_name, sae_lambda_sparse)

            if pretrained_sae_layers_string != "":

                for name in self.pretrained_sae_layers_list:
                    setattr(self, f"sae_{name}_inp_size", GetSaeInpSize(self.model, name, self.train_dataloader, self.device, self.model_name).get_sae_inp_size())
                # if we use pretrained SAEs, we load all of them
                for pretrain_sae_name in self.pretrained_saes_list:
                    # the last layer of the pretrain_sae_name is the layer on which the SAE was trained
                    # for example, "fc1&fc2&fc3" --> pretrain_sae_layer_name = "fc3"
                    pretrain_sae_layer_name = pretrain_sae_name.split("&")[-1]
                    temp_sae_inp_size = getattr(self, f"sae_{pretrain_sae_layer_name}_inp_size")
                    setattr(self, f"model_sae_{pretrain_sae_name}", load_pretrained_model(sae_model_name,
                                                                                        temp_sae_inp_size,
                                                                                        self.sae_weights_folder_path,
                                                                                        sae_expansion_factor=sae_expansion_factor,
                                                                                        layer_name=pretrain_sae_name,
                                                                                        params=self.params_string))
                    sae_model = getattr(self, f"model_sae_{pretrain_sae_name}")
                    sae_model = sae_model.to(self.device)
                    sae_model = sae_model.eval()
                    for param in sae_model.parameters():
                        param.requires_grad = False
                    print("Loaded pretrained SAE model on layer", pretrain_sae_name)
                                
            # we instantiate a fresh SAE for the layer on which we want to train the SAE
            if self.training:    
                setattr(self, f"sae_{self.train_sae_layer}_inp_size", GetSaeInpSize(self.model, self.train_sae_layer, self.train_dataloader, self.device, self.model_name).get_sae_inp_size())
                temp_sae_inp_size = getattr(self, f"sae_{self.train_sae_layer}_inp_size")
                self.sae_model = load_model(sae_model_name, img_size=temp_sae_inp_size, expansion_factor=sae_expansion_factor).to(self.device)
                self.sae_optimizer, _ = get_optimizer(sae_optimizer_name, self.sae_model, sae_learning_rate)

            # if we are using an SAE, we also create a copy of the original model so that we have 2 models
            # one modified model (with SAE) + one original model --> enables us to compare the outputs of those models
            # This model is always used in inference mode only
            self.model_copy = copy.deepcopy(self.model) # using load_pretrained_model might not give exactly the same model!
            self.model_copy = self.model_copy.to(self.device)
            self.model_copy.eval()
            for param in self.model_copy.parameters():
                param.requires_grad = False


# IN THE HOOK FUNCTION

   # use the sae to modify the output of the specified layer of the original model
        if use_sae and name in self.layer_names.split("&"):
            # get the SAE model corresponding to the current layer
            if self.training and name==self.train_sae_layer:
                # if self.training==True --> self.train_sae_layer and self.sae_model were defined
                sae_model = self.sae_model
            else:
                # if we use pretrained SAE on the given layer, we load it
                for entry in self.pretrained_saes_list:
                    if entry.endswith(name):
                        sae_model = getattr(self, f"model_sae_{entry}")