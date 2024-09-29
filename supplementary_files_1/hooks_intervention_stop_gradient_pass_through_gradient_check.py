'''
Code to check that stop-gradient and pass-through gradient work as expected.

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
    - 3) The gradient of x_hat and x_d should be the same.
    Pass-through gradient related
    - 4) The gradients of x_u and x_d should be different (it would be highly unlikely if they were the same)
    - 5) The gradient in a model without intervention/SAE should be equal to that of x_d but not equal to that of x_u
    - 6) To ensure that the gradient of x_u is correctly backpropagated to the input: The gradient wrt input is different in a model without intervention/SAE and in a model with intervention/SAE but without pass-through gradient

    
Experiment 3): With stop gradient and with pass-through gradient
    - 1) The gradient wrt x_d should be the same as the gradient wrt x_u
    - 2) To ensure that the gradient of x_u is correctly backpropagated to the input: The gradient wrt input is the same as in a model without intervention/SAE


Experiment 4): Without stop gradient and with pass-through gradient
    Pass-through gradient is only required if we use a stop-gradient. Otherwise, the gradient flows back as usual anyways. 
    Hence, there are no experiments for this setting.
'''

import torch
import torch.nn as nn
import copy

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
    def __init__(self, in_features, hidden_features, out_features):
        super(OriginalModel, self).__init__()
        self.layer1 = nn.Linear(in_features, hidden_features)
        self.layer2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.layer1(x)
        return self.layer2(x)
    
in_features = 4
hidden_features = 2
sae_in_features = hidden_features
sae_hidden_features = 5
out_features = 1

sae = SAE(sae_in_features, sae_hidden_features)
model = OriginalModel(in_features, hidden_features, out_features)
model_copy = copy.deepcopy(model) # we use deepcopy rather than a new instance because then the weights would be different
# Note: the output of model and model_copy should always be the same because even when using an SAE we pass back reconstruction + error

num_samples = 1
input_data = torch.randn(num_samples, in_features, requires_grad=True)
target_data = torch.randint(0, out_features, (num_samples,1), dtype=torch.float)
criterion = nn.MSELoss()

#################################################################

'''
class InterventionWithPassThroughGradient(autograd.Function):
    @staticmethod
    def forward(ctx, x_u, x_d):
        # x_u is original value (upstream), x_d is intervened value (downstream)
        ctx.save_for_backward(x_u, x_d)
        return x_d

    @staticmethod
    def backward(ctx, grad_output):
        x_u, x_d = ctx.saved_tensors
        grad_x_u = grad_output
        print(grad_output)
        grad_x_d = grad_output.clone()  # Gradient w.r.t the input x
        return grad_x_u, grad_x_d
'''
    
    
class Experiments:
    def __init__(self, model, sae, input_data, target_data, criterion, model_copy=None):
        self.model = model
        self.model_copy = model_copy 
        self.sae = sae
        self.input_data = input_data
        self.input_data_hook = self.input_data.register_hook(lambda grad: setattr(self, 'grad_input_data', grad))
        self.target_data = target_data
        self.criterion = criterion

    def intervention_hook(self, module, input, output, sae, use_stop_grad, use_pass_through_grad):
        # input is the input into layer1, i.e., the input data. Output is the output of layer1, and thus
        # the quantity we want to intervene on
        x_u = output # u stands for upstream
        encoded, x_hat = sae(x_u)
        sae_error = x_u - x_hat
        
        if use_stop_grad:
            # Apply stop gradient: x_d = x_hat + stopgrad(x_u - x_hat), d stands for "downstream"
            x_d = x_hat + sae_error.detach()
        else:
            x_d = x_hat + sae_error
            
        '''
        if use_pass_through_grad:
            intervention_with_pass_through_gradient = InterventionWithPassThroughGradient.apply
            x_intervened = intervention_with_pass_through_gradient(x_u, x_d)
        else:
            x_intervened = x_d
        '''

        self.x_hat = x_hat
        self.x_d = x_d
        self.x_u = x_u
        self.encoded = encoded
        self.sae_error = sae_error
            
        # Store the gradient of the loss wrt the different quantities
        # We use: https://pytorch.org/docs/stable/generated/torch.Tensor.register_hook.html
        # i.e. we register a backward hook on a tensor (and not a module f.e.)
        # Signature: hook(grad) -> Tensor or None. Thus, the input is the gradient flowing through this tensor,
        # and either we return a new gradient that will be used or nothing. 
        self.x_hat_hook = x_hat.register_hook(lambda grad: setattr(self, 'grad_x_hat', grad)) 
        self.x_d_hook = x_d.register_hook(lambda grad: setattr(self, 'grad_x_d', grad))
        #if use_pass_through_grad:
        #    self.x_u_hook_pass_through = x_u.register_hook(lambda grad: self.grad_x_d)
        self.x_u_hook = x_u.register_hook(lambda grad: setattr(self, 'grad_x_u', grad))
        self.encoded_hook = encoded.register_hook(lambda grad: setattr(self, 'grad_encoded', grad))
        self.sae_error_hook = sae_error.register_hook(lambda grad: setattr(self, 'grad_sae_error', grad))

        return x_d
    
    def encoder_backward_hook(self, module, grad_input, grad_output):
        '''
        self.encoder_grad = grad_output[0]  # Save the gradient w.r.t. encoder output
        # if encoder_grad is not zero vector
        if torch.allclose(self.encoder_grad, torch.zeros_like(self.encoder_grad)):
            # if we don't apply a stop gradient, the gradient wrt encoder output should be close to zero
            print("The stop gradient does not seem to work.")
        else:
            print("The stop gradient seems to work.")
        print(f"Encoder Gradient: {self.encoder_grad}")	
        '''
        pass

    def register_hooks(self, model, use_stop_grad, use_pass_through_grad, sae=None):

        # if an SAE is specified, we apply the intervention_hook
        if sae is not None:
            hook = model.layer1.register_forward_hook(
                lambda module, input, output: self.intervention_hook(module, input, output, sae, use_stop_grad, use_pass_through_grad)
            )
        else: 
            # we don't need a special hook function
            # grad_input is the gradient wrt the input of layer1, i.e., wrt the input data, while grad_output is the gradient
            # wrt the output of layer1, which is what we're interested in
            hook = model.layer1.register_full_backward_hook(lambda module, grad_input, grad_output: setattr(self, 'grad_x_u_original', grad_output))

        # model.layer1.register_hook(lambda grad: print(grad)) register hook only works on tensors but not on modules
        hook1 = model.layer1.register_full_backward_hook(lambda module, grad_input, grad_output: grad_input) #hook_pass_through(module, grad_input, grad_output)) #print(grad_output, grad_input)) #setattr(self, 'grad_x_u_original', grad_output))

        # Register backward hook
        #sae.encoder.register_full_backward_hook(lambda module, grad_input, grad_output: self.encoder_backward_hook(module, grad_input, grad_output))
                                                
        return hook, hook1
    
    def perform_experiment(self, use_stop_grad, use_pass_through_grad, experiment_id):
        hook, hook1 = self.register_hooks(self.model, use_stop_grad, use_pass_through_grad, sae=self.sae)
        output = self.model(self.input_data)
        loss = self.criterion(output, self.target_data)
        loss.backward()

        print("Gradient wrt x_hat:".ljust(30), self.grad_x_hat)
        print("Gradient wrt x_d:".ljust(30), self.grad_x_d)
        print("Gradient wrt x_u:".ljust(30), self.grad_x_u)
        print("Gradient wrt encoded:".ljust(30), self.grad_encoded)
        if hasattr(self, 'grad_sae_error'):
            print("Gradient wrt sae_error:".ljust(30), self.grad_sae_error)
        else:
            # if we apply stop gradient, sae_error has no gradient!
            print("Gradient wrt sae_error:".ljust(30), "None") 
        print("Gradient wrt input data:".ljust(30), self.grad_input_data)
        grad_input_data_sae_model = self.grad_input_data.clone()
        print("input data grad:", self.input_data.grad)

        hook.remove()
        hook1.remove()
        self.x_hat_hook.remove()
        self.x_d_hook.remove()
        self.x_u_hook.remove()
        self.encoded_hook.remove()
        if use_pass_through_grad:
            self.x_u_hook_pass_through.remove()
        self.sae_error_hook.remove()
        self.input_data_hook.remove()

        print("--------------------")
        if experiment_id == "2":
            # we use no sae here --> no intervention
            hook, hook1 = self.register_hooks(self.model_copy, use_stop_grad, use_pass_through_grad)
            output_2 = self.model_copy(self.input_data)
            loss_2 = self.criterion(output_2, self.target_data)
            loss_2.backward()
            hook.remove()
            hook1.remove()
            print("Gradient wrt original x_u:".ljust(30), self.grad_x_u_original[0]) # gradient is a tuple, we only want the first element
            print("Gradient wrt input data in model without intervention/SAE:".ljust(30), self.grad_input_data)
            print("input data grad in model without intervention/SAE:", self.input_data.grad)


        ###################### CHECK IF EXPERIMENT WORKED AS EXPECTED ######################
        if experiment_id == "1":
            cond1 = torch.allclose(self.grad_x_hat, torch.zeros_like(self.grad_x_hat))
            cond2 = torch.allclose(self.grad_encoded, torch.zeros_like(self.grad_encoded))
            if cond1 and cond2:
                print("Success")
            else:
                print("Failure")

        elif experiment_id == "2":
            cond1 = not hasattr(self, 'grad_sae_error')
            cond2 = torch.allclose(self.grad_x_d @ self.sae.decoder.weight, self.grad_encoded, atol=1e-2)
            # if cond2 is true, this indicates that the encoder gradient seems to be correct
            # we have to set quite a big tolerance here, since there is some discrepancy between the two gradients, such as 0.0005,
            # this is apparently due to some numerical issues (but based on what I saw it was 0.0005 or -0.0005 for all values, so it
            # seems to be consistent at least)
            # the first quantity is the analytical gradient of the loss wrt the encoder output based on the chain rule
            # the second quantity is the gradient of the loss wrt the encoder output based on the computed values and the backward hook
            cond3 = torch.allclose(self.grad_x_d, self.grad_x_hat)
            cond4 = not torch.allclose(self.grad_x_u, self.grad_x_d)
            cond5 = torch.allclose(self.grad_x_u_original[0], self.grad_x_d)
            # if grad_x_d == grad_x_u_original and grad_x_d != grad_x_u, this implies that grad_x_u != grad_x_u_original
            cond6 = not torch.allclose(self.grad_input_data, grad_input_data_sae_model)
            if cond1 and cond2 and cond3 and cond4 and cond5 and cond6:
                print("Success")
            else:
                print("Failure")

        elif experiment_id == "3":
            cond1 = torch.allclose(self.grad_x_d, self.grad_x_u)
            if cond1:
                print("Success")
            else:
                print("Failure")
    

#################################################################

### 1) Original model + SAE/intervention
'''
print("Intervention without stop gradient and without pass-through gradient.")
use_stop_grad = False
use_pass_through_grad = False
experiment_id = "1"
experiments = Experiments(model, sae, input_data, target_data, criterion)
experiments.perform_experiment(use_stop_grad, use_pass_through_grad, experiment_id)
'''

### 2) Original model + SAE/intervention + stop-gradient
#'''
print("-----------------------------------------------------------------")
print("Intervention with stop gradient and without pass-through gradient.")
use_stop_grad = True
use_pass_through_grad = False
experiment_id = "2"
experiments = Experiments(model, sae, input_data, target_data, criterion, model_copy=model_copy)
experiments.perform_experiment(use_stop_grad, use_pass_through_grad, experiment_id)
#'''

### 3) Original model + SAE/intervention + stop-gradient + pass-through gradient
'''
print("-----------------------------------------------------------------")
print("Intervention with stop gradient and with pass-through gradient.")
use_stop_grad = True
use_pass_through_grad = True
experiment_id = "3"
experiments = Experiments(model, sae, input_data, target_data, criterion)
experiments.perform_experiment(use_stop_grad, use_pass_through_grad, experiment_id)
'''

### DO SOME TECHNICAL CHECKS WHETHER ALL DIFFERENT TYPES OF HOOKS GIVE THE SAME GRADIENT VALUES!!!!
### CLEAN UP CODE!!!