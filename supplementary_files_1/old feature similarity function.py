# OLD FEATURE SIMILARITY FUNCTION

def feature_similarity(model,activations,device):
    # REVISE THIS FUNCTION, SEEMS A BIT WEIRD, F.E. WHY DO WE ITERATE OVER MODULE NAMES AND NOT JUST THE KEYS OF ACTIVATIONS?
    # we calculate the feature similarity between the modified and original for one batch
    activation_similarity = {}
    module_names = get_module_names(model)
    for name_1 in module_names:
        activations = [(k,v) for k, v in activations.items() if k[0] == name_1]
        # activations should be of the form (with len(activations)=3)
        # [((name_1, 'modified'), [list of tensors, one for each batch]), 
        #  ((name_1, 'original'), [list of tensors, one for each batch])]
        # and if we inserted an SAE at the given layer with name_1, then 
        # ((name_1, 'sae'), [list of tensors, one for each batch]) is the first entry of
        # activations
        # we check whether activations has the expected shape
        if (activations[-2][0][1] == "modified" and activations[-1][0][1] == "original"):
            activation_list_1 = activations[-1][1]
            activation_list_2 = activations[-2][1]

            # we check whether the length of both activation lists corresponds to the number of batches
            if len(activation_list_1) != 1 or len(activation_list_2) != 1:
                raise ValueError(f"For layer {name_1}: The length of the activation lists for computing feature similarity (length of activation list of modified model {len(activation_list_2)}, original model {len(activation_list_1)}) should be 1 since we consider 1 batch.")

            dist_mean = 0.0
            dist_std = 0.0
            for act1, act2 in zip(activation_list_1, activation_list_2):
                activation_1 = act1.to(device)
                activation_2 = act2.to(device)
                # dist is the distance between each pair of samples --> dimension is [batch_size]
                sample_dist = torch.linalg.norm(activation_1 - activation_2, dim=1)
                dist_mean += sample_dist.mean().item()
                dist_std += sample_dist.std().item()   
            activation_similarity[name_1] = (dist_mean, dist_std)
        else:
            raise ValueError("Activations has the wrong shape for evaluating feature similarity.")
    return activation_similarity