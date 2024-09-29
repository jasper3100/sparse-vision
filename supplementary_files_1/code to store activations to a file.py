# CODE TO STORE ACTIVATIONS TO A FILE
# in model_pipeline at the very bottom

# requires storing activations over a longer horizon than just one batch
# as done currently in the code (02.03.2024)

# store the feature maps (of up to those 3 types: of original model, modified model and encoder output)
# after the last epoch 
if self.store_activations:
    self.activations = {k: torch.cat(v, dim=0) for k, v in self.activations.items()}
    store_feature_maps(self.activations, self.activations_folder_path, params=params)
    print(f"Successfully stored activations.")
    # Previous code to store activations:
    #for name in self.activations.keys():
    #    self.activations[name] = torch.cat(self.activations[name], dim=0)
    #store_feature_maps(self.activations, self.activations_folder_path, params=params)