'''
INTRO TO USING NNSIGHT
'''

from collections import OrderedDict
import torch

input_size = 5
hidden_dims = 10
output_size = 2

net = torch.nn.Sequential(
    OrderedDict(
        [
            ("layer1", torch.nn.Linear(input_size, hidden_dims)),
            ("layer2", torch.nn.Linear(hidden_dims, output_size)),
        ]
    )
)#.requires_grad_(False)

num_samples = 1
input = torch.rand((num_samples, input_size))
input_2 = torch.rand((num_samples, input_size))

from nnsight import NNsight

model = NNsight(net)

# in the tracing context (i.e. in the indented space after "with model.trace(input)"
# we specify how the model should run
# the model is run upon exiting the tracing context
with model.trace(input):#with model.trace() as tracer:

    # sometimes we might want to see what the output of a model would be
    # with and without intervention. We could run two tracing contexts, requiring
    # two forward passes. But there is a better solution, namely to to use tracer.invoke 
    # then, the inputs of the different invokers will be batched together and executed in one forward pass!
    # ("Batching")
    #with tracer.invoke(input) as invoker:

    # we call .save() if we want to use the value outside of the tracing context
    l2_input = model.layer2.input.save()
    #l1_amax = torch.argmax(l1_output, dim=1).save()

    print(f"layer1 output shape: {model.layer1.output.shape}")

    # save the output before the edit to compare
    # use .clone() before saving as the setting operation is in-place
    l1_output_before = model.layer1.output.clone().save()

    # set the first element of the output of layer1 to 0, the first dimension is the batch dimension
    model.layer1.output[:, 0] = 0

    l1_output_after = model.layer1.output.save()

    #'''

    layer1_grad = model.layer1.output.grad.clone().save()
    layer1_grad_detach = model.layer1.output.grad.detach().clone().save()
    model.layer1.output[:, 0] = 1.0
    layer1_grad_after = model.layer1.output.grad.save()
    loss = model.output.sum()
    loss.backward()



        # in order to have a grad we need a loss for backpropagation
        #'''
with model.trace(input):
    print(f"layer1 output shape: {model.layer1.output.shape}")
    l1_output_before_original = model.layer1.output.save()

        

    #layer1_output_grad_original = model.layer1.output.grad.save()

    #loss = model.output.sum()
    #loss.backward()


print(l1_output_before)
print(l1_output_after)
print(l2_input)
# the shape of values from .input is in the form: tuple(tuple(args), dictionary(kwargs))
# tuple(args) = tuple of all positional arguments (i.e. arguments are determined based on their position, f.e. f(1,2,3))
# dictionary(kwargs) = dictionary of all keyword arguments (i.e. arguments are determined based on their keyword/name, f.e. f(c=3,b=2,a=1))
#print(l1_amax)
#print(layer1_output_grad_original)
print(l1_output_before_original)

# the below shows that storing gradients is not easily possible with .save()
# because it is modified in-place
print(f"layer1_grad: {layer1_grad}")
print(f"layer1_grad_detach: {layer1_grad_detach}")
print(f"layer1_grad_after: {layer1_grad_after}")
