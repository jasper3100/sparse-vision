import torch.nn.functional as F
import torch

# CHECK THAT CROSS ENTROPY LOSS WORKS AS EXPECTED

'''
By the definition of torch.nn.CrossEntropyLoss(), the input tensor should contain the 
unnormalized logits for each class, i.e, the raw output of the model, while the target
tensor should contain either class indices (f.e. [0,0,1]) or probabilities for each class.
Here, we choose the latter.
'''

# Do some testing
input = torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
target = torch.tensor([[3.0, 2.0, 1.0], [1.0, 2.0, 3.0]])

print(input.shape) 
# --> same format as model output

target = F.softmax(target, dim=1)
print(target)
# --> seems to do what it should

def manual_cross_entropy(input, target):
    # where reduction "mean", i.e., we take the average loss
    # weights = None, i.e., each weight w_c = 1, i.e., each class has the same importance weight
    return -1/(input.size(0)) * torch.sum(torch.log(F.softmax(input, dim=1)) * target)

loss = F.cross_entropy(input, target)
print(loss)

loss1 = manual_cross_entropy(input, target)
print(loss1)
# --> both losses are the same

print(F.cross_entropy(target, input))
# --> order of input arguments matters, as expected!