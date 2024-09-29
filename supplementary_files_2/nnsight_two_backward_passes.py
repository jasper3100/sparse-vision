# Doing 2 backward passes and checking which gradients are recorded

import torch
import torch.nn as nn
from nnsight import NNsight

torch.manual_seed(0)

########################################################################

class SAE(nn.Module):
    def __init__(self, in_features, hidden_features):
        super(SAE, self).__init__()
        self.encoder = nn.Linear(in_features, hidden_features)
        self.decoder = nn.Linear(hidden_features, in_features)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

class OriginalModel(nn.Module):
    def __init__(self, in_features, out_features):
        super(OriginalModel, self).__init__()
        self.layer1 = nn.Linear(in_features, 4)
        self.layer2 = nn.Linear(4, 3)
        self.layer3 = nn.Linear(3, 2)
        self.layer4 = nn.Linear(2, out_features)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return self.layer3(x)

in_features = 5
out_features = 2
sae_layer1 = SAE(4, 6) # in_features, hidden_features
sae_layer2 = SAE(3, 5)
sae_layer3 = SAE(2, 4)
model = OriginalModel(in_features, out_features)
model = NNsight(model)

num_samples = 1
input_data = torch.randn(num_samples, in_features, requires_grad=True)
target_data = torch.randint(0, 5, (num_samples,out_features), dtype=torch.float)
criterion = nn.MSELoss()

########################################################################
        
def intervention(model, sae, layer, layer_grad_original=None, pass_through=True):
    encoder_output, decoder_output = sae(getattr(model, layer).output)
    sae_error = (getattr(model, layer).output - decoder_output).detach() # stop gradient 
    getattr(model, layer).output = decoder_output + sae_error # intervention
    if pass_through:
        getattr(model, layer).output.grad[:] = layer_grad_original

    # saving quantities
    encoder_output_grad = encoder_output.grad.save()
    encoder_output = encoder_output.save()
    layer_grad = getattr(model, layer).output.grad.save()
    decoder_output_grad = decoder_output.grad.save()

    return encoder_output_grad, encoder_output, layer_grad, decoder_output_grad

########################################################################

# get instance of intervention
validation = True # set to True during debugging

#####################################
# Model without intervention, backprop from L2 loss
with model.trace(input_data, validate=validation):
    layer1_grad_original = model.layer1.output.grad.save()
    layer2_grad_original = model.layer2.output.grad.save()
    layer3_grad_original = model.layer3.output.grad.save()
    criterion(model.output, target_data).backward()

print("Model without intervention, backprop from L2 loss")
print("grad_l1_original", layer1_grad_original)
print("-----------------------------------------------")

#####################################
# Model without intervention, backprop from L1 loss
with model.trace(input_data, validate=validation):
    layer1_grad_l1loss = model.layer1.output.grad.save()
    nn.L1Loss()(model.output, target_data).backward()

print("Model without intervention, backprop from L1 loss")
print("grad_l1_l1loss", layer1_grad_l1loss)
print("--> As expected there is a difference to the gradient from above")
print("-----------------------------------------------")

################################
# Doing backprop from model loss, with L1 and L2 loss
with model.trace(input_data, validate=validation):
    layer1_grad = model.layer1.output.grad.save()
    loss_v6_2 = nn.L1Loss()(model.output, target_data).save()
    loss_v6_1 = criterion(model.output, target_data).save()


    loss_v6_1.backward(retain_graph=True)
    loss_v6_2.backward(retain_graph=True) 

    #loss_v6_2.backward(retain_graph=True)
    #loss_v6_1.backward(retain_graph=True) # not sure why I need retain_graph here???

print("Backprop from L1 and L2")
print("grad_l1", layer1_grad)
print("--> The gradients wrt that loss are recorded, that is computed first (the loss that is computed first not the backward!!!)")
print("-----------------------------------------------")


######################################
# Doing backprop from encoder output 2 
# In this case, we shouldn't use the original gradient that was computed backproping from model loss
# as pass-through gradient. Because then the gradient wrt the encoder output would also refer to model loss
# Instead, we should disable pass-through gradient of layer 1 and layer 2.
with model.trace(input_data, validate=validation):
    encoder_output_grad_l1_v1, encoder_output_l1_v1, grad_l1_v1, decoder_output_grad_l1_v1 = intervention(model, sae_layer1, "layer1", pass_through=False)
    encoder_output_grad_l2_v1, encoder_output_l2_v1, _, _ = intervention(model, sae_layer2, "layer2", pass_through=False)

    loss_v1 = torch.sum(encoder_output_l2_v1).save()
    loss_v1.backward() 

print("Doing backprop from encoder output 2")
print("encoder_output_grad_l1_v1", encoder_output_grad_l1_v1)
print("grad_l1_v1", grad_l1_v1)
assert torch.allclose(decoder_output_grad_l1_v1, grad_l1_v1)
print("loss value v1", loss_v1)
print("------------------------------------------------")

################################
# Doing backprop from model loss
with model.trace(input_data, validate=validation):
    encoder_output_grad_l1_v2, encoder_output_l1_v2, grad_l1_v2, decoder_output_grad_l1_v2 = intervention(model, sae_layer1, "layer1", layer_grad_original=layer1_grad_original)
    encoder_output_grad_l2_v2, encoder_output_l2_v2, _, _ = intervention(model, sae_layer2, "layer2", layer_grad_original=layer2_grad_original)
    _, _, grad_l3_v2, _ = intervention(model, sae_layer3, "layer3", layer_grad_original=layer3_grad_original)

    loss_v2 = criterion(model.output, target_data).save()
    loss_v2.backward()

print("Doing backprop from model loss")
print("encoder_output_grad_l1_v2", encoder_output_grad_l1_v2)
print("grad_l1_v2", grad_l1_v2)
print("grad_l3_v2", grad_l3_v2)
assert torch.allclose(decoder_output_grad_l1_v2, grad_l1_v2)
print("loss value v2", loss_v2)
print("--> As expected the gradients are different from the partial backward.")
print("------------------------------------------------")

################################
# Doing backprop from model loss, without pass-through gradient
with model.trace(input_data, validate=validation):
    encoder_output_grad_l1_v4, encoder_output_l1_v4, grad_l1_v4, decoder_output_grad_l1_v4 = intervention(model, sae_layer1, "layer1", pass_through=False)
    encoder_output_grad_l2_v4, encoder_output_l2_v4, _, _ = intervention(model, sae_layer2, "layer2", pass_through=False)

    loss_v4 = criterion(model.output, target_data).save()
    loss_v4.backward()

print("Doing backprop from model loss, without pass-through gradient")
print("encoder_output_grad_l1_v4", encoder_output_grad_l1_v4)
print("grad_l1_v4", grad_l1_v4)
assert torch.allclose(decoder_output_grad_l1_v4, grad_l1_v4)
print("loss value v4", loss_v4)
print("--> As expected the gradients are different than with pass-through gradient, and different from the partial backward.")
print("------------------------------------------------")


###############################
# Doing backprop from 2 places, from model loss and encoder output
# Since I want to see if the partial backprop still works, even when applying the full backprop afterwards.
# I also disable the pass-through gradient on layers 1 and 2 here (otherwise the partial backprop wouldnt work to start with)
with model.trace(input_data, validate=validation):
    encoder_output_grad_l1_v3, encoder_output_l1_v3, grad_l1_v3, decoder_output_grad_l1_v3 = intervention(model, sae_layer1, "layer1", pass_through=False)
    encoder_output_grad_l2_v3, encoder_output_l2_v3, _, _ = intervention(model, sae_layer2, "layer2", pass_through=False)
    _, _, grad_l3_v3, _ = intervention(model, sae_layer3, "layer3", layer_grad_original=layer3_grad_original)

    loss_v3_2 = torch.sum(encoder_output_l2_v3).save()
    loss_v3_1 = criterion(model.output, target_data).save()

    loss_v3_1.backward(retain_graph=True)
    loss_v3_2.backward(retain_graph=True) # need retain_graph=True here

    #loss_v3_2.backward(retain_graph=True)
    #loss_v3_1.backward() # don't need retain_graph here, and using it doesn't impact the results

print("Doing backprop from 2 places, from model loss and encoder output")
print("encoder_output_grad_l1_v3", encoder_output_grad_l1_v3)
print("grad_l1_v3", grad_l1_v3)
print("grad_l3_v3", grad_l3_v3)
assert torch.allclose(decoder_output_grad_l1_v3, grad_l1_v3)
print("loss value v3_1", loss_v3_1)
print("loss value v3_2", loss_v3_2)
print("--> In both cases, the gradient wrt encoder output layer 1 corresponds to those of the partial backward. Hence,")
print("    when doing a full backward pass and then a partial backward pass, the gradients are overwritten. But when doing a partial")
print("    backward pass and then a full backward pass, the gradients are not overwritten...")
print("--> In both cases, the gradient wrt layer 3 is computed and corresponds to the gradient computed during the full backward pass.")
print("    (For the partial backward, this gradient isn't computed.)")
print("------------------------------------------------")

################################
# Doing backprop from encoder output layer 3
# We should use pass through gradient on layers 1,2,3 here so that the gradient can flow through.
with model.trace(input_data, validate=validation):
    encoder_output_grad_l1_v4, encoder_output_l1_v4, grad_l1_v4, decoder_output_grad_l1_v4 = intervention(model, sae_layer1, "layer1", pass_through=False)
    encoder_output_grad_l2_v4, encoder_output_l2_v4, grad_l2_v4, decoder_output_grad_l2_v4 = intervention(model, sae_layer2, "layer2", pass_through=False)
    encoder_output_grad_l3_v4, encoder_output_l3_v4, grad_l3_v4, decoder_output_grad_l3_v4 = intervention(model, sae_layer3, "layer3", pass_through=False)

    loss_v4 = torch.sum(encoder_output_l3_v4).save()
    loss_v4.backward()

print("Doing backprop from encoder output layer 3")
print("encoder_output_grad_l1_v4", encoder_output_grad_l1_v4)
print("grad_l1_v4", grad_l1_v4)
print("-----------------------------------------------")

################################
# Doing backprop from 2 places, from encoder output layer 2 and encoder output layer 3
with model.trace(input_data, validate=validation):
    encoder_output_grad_l1_v5, encoder_output_l1_v5, grad_l1_v5, decoder_output_grad_l1_v5 = intervention(model, sae_layer1, "layer1", pass_through=False)
    encoder_output_grad_l2_v5, encoder_output_l2_v5, grad_l2_v5, decoder_output_grad_l2_v5 = intervention(model, sae_layer2, "layer2", pass_through=False)
    encoder_output_grad_l3_v5, encoder_output_l3_v5, grad_l3_v5, decoder_output_grad_l3_v5 = intervention(model, sae_layer3, "layer3", pass_through=False)

    loss_v5_1 = torch.sum(encoder_output_l2_v5).save()
    loss_v5_2 = torch.sum(encoder_output_l3_v5).save()
    
    #loss_v5_1.backward(retain_graph=True)
    #loss_v5_2.backward()

    loss_v5_2.backward(retain_graph=True)
    loss_v5_1.backward(retain_graph=True) # need retain_graph here

print("Doing backprop from 2 places, from encoder output layer 2 and encoder output layer 3")
print("encoder_output_grad_l1_v5", encoder_output_grad_l1_v5)
print("grad_l1_v5", grad_l1_v5)
print("--> In both cases, the gradients correspond to those of the earlier backward pass (on layer 2 encoder output)")
print("-----------------------------------------------")



# --> It seems like the gradients of the backward call of an earlier layer are recorded
# Even though it seems weird, the decoder output is earlier because it replaces the model output,
# and the encoder output depends on the model output

# Case 3
#criterion(model_output_v3, target_data).backward(retain_graph=True)
#nn.L1Loss()(model_output_v3, target_data).backward()

#nn.L1Loss()(model_output_v3, target_data).backward(retain_graph=True)
#criterion(model_output_v3, target_data).backward()

# --> In both cases, the gradients of the first backward are recorded!!!

