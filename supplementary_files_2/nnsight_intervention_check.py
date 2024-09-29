'''
Code to check that stop-gradient and pass-through gradient work as expected using nnsight.

Glossary:
x_u = upstream activation / original output of layer
x_hat = SAE reconstruction
x_d = downstream activation / intervened output of layer = x_hat + sae_error, where sae_error = x_u - x_hat

Experiment 1): Without stop gradient and without pass-through gradient. 
    - 1) The gradient wrt the encoder output should be zero. 
            This can be shown mathematically (see Marks et al. paper). But intuitively it makes sense as well:
            We have x_d = x_hat + sae_error = x_hat + x_u - x_hat = x_u. So if we don't restrict gradient flow in 
            any way, the gradient has no reason to flow through x_hat (and thus through encoded), and it doesn't!
            Hence, we need to stop the gradient between sae_error and x_u. This is achieved by the stop-gradient on 
            the sae_error.
    - 2) By the above, the gradient wrt x_hat should be zero.
    - 3) The output of the model with intervention should be the same as the output of the model without intervention.

    
Experiment 2): With stop gradient and without pass-through gradient
    Stop-gradient related
    - 1) The sae error should not have a gradient
    - 2) The gradient wrt the encoder output should be equivalent to the gradient wrt layer output times SAE features (by chain rule)
    - 3) The gradient of x_hat and x_d should be the same by chain rule with z = x + y -> dL/dz = dL/dx = dL/dy
    - 4) For layer2, the gradient after the intervention is the same as the original gradient because the gradient is not
        modified between layer2 output and model output.
    - 5) For layer1, the gradient after the intervention is different than the original gradient because the gradient is
        modified between layer1 output and layer2 because we don't use the pass-through gradient (in layer2).


Experiment 3): With stop gradient and with pass-through gradient
    - 1) The gradient wrt layer output is the same as the original gradient wrt layer output.


Experiment 4): Without stop gradient and with pass-through gradient
    Pass-through gradient is only required if we use a stop-gradient. Otherwise, the gradient flows back as usual anyways. 
    Hence, there are no experiments for this setting.
'''

import torch
import torch.nn as nn
from nnsight import NNsight
from einops import rearrange

torch.manual_seed(0)

###################### MODELS AND DATA (BORING STUFF) ######################
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
        self.layer3 = nn.Linear(3, out_features)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return self.layer3(x)
    
in_features = 5
out_features = 2
sae_layer1 = SAE(4, 6) # in_features, hidden_features
sae_layer2 = SAE(3, 5)
model = OriginalModel(in_features, out_features)
model = NNsight(model)

num_samples = 1
input_data = torch.randn(num_samples, in_features, requires_grad=True)
target_data = torch.randint(0, 5, (num_samples,out_features), dtype=torch.float)
criterion = nn.MSELoss()

#################################################################

def experiments(model, input_data, target_data, criterion, sae_layer1, sae_layer2, stop_gradient, pass_through_gradient, experiment_id): 

    validation = False # this is set to True in the original code
    # set to True, during debugging, because “Validating” mean trying 
    # to execute the intervention proxies with “fake” inputs to see if they work

    ### MODEL WITHOUT INTERVENTION
    with model.trace(input_data, validate=validation):
        # we record the gradients in the model without interventions because it seems like
        # despite detaching gradients, they are still modified by the intervention
        layer1_grad_original = model.layer1.output.grad.save()
        layer2_grad_original = model.layer2.output.grad.save()
        layer1_output_original = model.layer1.output.save()
        layer2_output_original = model.layer2.output.save()
        model_output_original = model.output.save()
        criterion(model.output, target_data).backward() # so that we have gradients

    ### MODEL WITH INTERVENTION
    with model.trace(input_data, validate=validation):

        # record the gradients before the intervention
        # THIS DOESNT WORK AS PRINTED BELOW!!! SO THE ABOVE VERSION IS NEEDED!!!
        layer1_grad_before = model.layer1.output.grad.clone().save()
        layer2_grad_before = model.layer2.output.grad.clone().save()

        # compute the SAE outputs
        encoder_output_layer1, decoder_output_layer1 = sae_layer1(model.layer1.output)
        encoder_output_layer2, decoder_output_layer2 = sae_layer2(model.layer2.output)
        
        # Using rearrange seems to remove the gradient
        #a = encoder_output_layer1.shape[0]
        #b = encoder_output_layer1.shape[1]
        #encoder_output_layer1 = rearrange(encoder_output_layer1, 'a b -> (a b)')
        #encoder_output_layer1.grad = rearrange(encoder_output_layer1.grad, 'a b -> (a b)', a=a, b=b)
        #encoder_output_layer1 = rearrange(encoder_output_layer1, '(a b) -> a b', a=a, b=b)
        #encoder_output_layer1.grad = rearrange(encoder_output_layer1.grad, '(a b) -> a b', a=a, b=b)

        # record the SAE encoder output gradients
        sae_features_layer1_grad = encoder_output_layer1.grad.save()
        sae_features_layer2_grad = encoder_output_layer2.grad.save()

        sae_error_layer1 = model.layer1.output - decoder_output_layer1
        sae_error_layer2 = model.layer2.output - decoder_output_layer2

        if stop_gradient:
            # apply stop-gradient on the sae error
            sae_error_layer1 = sae_error_layer1.detach()
            sae_error_layer2 = sae_error_layer2.detach()    

        # intervene on the layer output 
        # --> sae features and errors are included in the computation graph
        # do not use [:] here because this would lead to some conflict related to the
        # computation graph
        model.layer1.output = decoder_output_layer1 + sae_error_layer1
        # or equivalently: setattr(model.layer1, 'output', decoder_output_layer1 + sae_error_layer1)
        # or equivalently: getattr(model, 'layer1').output = decoder_output_layer1 + sae_error_layer1
        model.layer2.output = decoder_output_layer2 + sae_error_layer2
        # apaarently the following two are equivalent: 
        # aux = getattr(model.layer1, 'output').save()
        # aux1 = getattr(model, 'layer1').output.save()       

        layer1_output = model.layer1.output.save()
        layer2_output = model.layer2.output.save()
        
        # record the gradients 
        layer1_grad = model.layer1.output.grad.save()
        layer2_grad = model.layer2.output.grad.save()
        # in fact, these are the same gradients as those of the SAE decoder output (x_hat) gradients
        # consequently, they are not the same as the gradients of the original layer output, which is verified below
        sae_decoder_output_layer1_grad = decoder_output_layer1.grad.save()
        sae_decoder_output_layer2_grad = decoder_output_layer2.grad.save()

        
        if pass_through_gradient:
            # implement pass-through gradient (i.e. reset the gradients to the original gradients)
            # including the [:] is important!
            model.layer1.output.grad[:] = layer1_grad_original
            # DOESNT WORK: model.layer1.output.grad = layer1_grad_original.clone()
            model.layer2.output.grad[:] = layer2_grad_original
        
        # record the model output
        model_output_sae = model.output.save()

        # propagate gradients back for sample 0, feature 0 of SAE on layer2
        #encoder_output_layer2[0, 0].backward()

        criterion(model.output, target_data).backward() # so that we have gradients

    
    if experiment_id == 1:
        print("\n-------------------- Experiment 1 --------------------\n")
        cond01 = torch.allclose(layer1_grad_before, layer1_grad_original)
        cond02 = torch.allclose(layer2_grad_before, layer2_grad_original)
        print("Measuring the gradient of a layer before intervention works in both ways: ", cond01 and cond02)    
        cond1 = torch.allclose(sae_features_layer1_grad, torch.zeros_like(sae_features_layer1_grad))
        cond2 = torch.allclose(sae_features_layer2_grad, torch.zeros_like(sae_features_layer2_grad))
        print("Without stop-gradient, the gradient wrt encoder output is zero as expected: ", cond1 and cond2)
        cond3 = torch.allclose(sae_decoder_output_layer1_grad, torch.zeros_like(sae_decoder_output_layer1_grad))
        cond4 = torch.allclose(sae_decoder_output_layer2_grad, torch.zeros_like(sae_decoder_output_layer2_grad))
        print("Gradient wrt decoder output is zero as expected: ", cond3 and cond4)
        cond5 = torch.allclose(model_output_sae, model_output_original)
        print("Output of SAE model is same as of original model as expected: ", cond5)

    elif experiment_id == 2:
        print("\n-------------------- Experiment 2 --------------------\n")
        cond01 = torch.allclose(layer1_grad_before, layer1_grad_original)
        cond02 = torch.allclose(layer2_grad_before, layer2_grad_original)
        print("Measuring the gradient of a layer before intervention works in both ways: ", cond01 and cond02)
        print("Trying to save the gradients of the sae errors yields an error, as expected: True") # I checked this
        cond1 = torch.allclose(sae_features_layer1_grad, layer1_grad @ sae_layer1.decoder.weight)
        cond2 = torch.allclose(sae_features_layer2_grad, layer2_grad @ sae_layer2.decoder.weight)
        print("The gradient of SAE features equals the gradient of the original layer times the decoder weights, as expected: ", cond1 and cond2)
        # one advantage compared to using hooks directly, was that there, apparently there were some small numerical
        # errors (in my implementation)
        cond3 = torch.allclose(layer1_grad, sae_decoder_output_layer1_grad)
        cond4 = torch.allclose(layer2_grad, sae_decoder_output_layer2_grad)
        print("The gradient wrt layer output (after intervention) is the same as the gradient wrt decoder output as expected: ", cond3 and cond4)
        cond5 = torch.allclose(layer2_grad, layer2_grad_original)
        print("For layer2, the gradient after the intervention is the same as the original gradient because \n the gradient is not modified between layer2 output and model output: ", cond5)
        cond6 = not torch.allclose(layer1_grad, layer1_grad_original)
        print("For layer1, the gradient after the intervention is different than the original gradient because \n the gradient is modified between layer1 output and layer2 because we don't use the pass-through gradient: ", cond6)
        
    elif experiment_id == 3:
        print("\n-------------------- Experiment 3 --------------------\n")
        cond01 = torch.allclose(layer1_grad_before, layer1_grad_original)
        cond02 = torch.allclose(layer2_grad_before, layer2_grad_original)
        print("Measuring the gradient of a layer before intervention works in both ways: ", cond01 and cond02)
        cond1 = torch.allclose(layer1_grad, layer1_grad_original)
        cond2 = torch.allclose(layer2_grad, layer2_grad_original)
        print("The gradient wrt layer output is the same as the original gradient, as expected: ", cond1 and cond2)


# Experiment 1
stop_gradient = False
pass_through_gradient = False
experiment_id = 1
experiments(model, input_data, target_data, criterion, sae_layer1, sae_layer2, stop_gradient, pass_through_gradient, experiment_id)

# Experiment 2
stop_gradient = True
pass_through_gradient = False
experiment_id = 2
experiments(model, input_data, target_data, criterion, sae_layer1, sae_layer2, stop_gradient, pass_through_gradient, experiment_id)

# Experiment 3
stop_gradient = True
pass_through_gradient = True
experiment_id = 3
experiments(model, input_data, target_data, criterion, sae_layer1, sae_layer2, stop_gradient, pass_through_gradient, experiment_id)