import torch

# Assuming activations is your tensor of shape (batch_size, num_neurons)
# For example, you might get it like this:
# activations = model(data)  # data is your input batch

# set a random seed
torch.manual_seed(0)

activations = torch.rand(4, 3)
print(activations)

batch_size, num_neurons = activations.shape

# Get the indices of the top-k activations for each neuron
topk_values, topk_indices = torch.topk(activations, k=2, dim=0)
print(topk_values)

# we add batch_idx*batch_size to every entry in the indices matrix to get the 
# index of the corresponding image in the dataset. For example, if the index in the
# batch is 2 and the batch index is 4 and batch size is 64, then the index of this 
# image in the dataset is 4*64 + 2= 258
batch_idx = 1
batch_size = 10
topk_indices += batch_idx*batch_size
print(topk_indices)

print("-----------------")
activations = torch.rand(4, 3)
print(activations)
topk_values_1, topk_indices_1 = torch.topk(activations, k=2, dim=0)

print(topk_values_1)
topk_indices_1 += batch_idx*20
print(topk_indices_1)

print("-----------------")
topk_values_merged = torch.cat((topk_values, topk_values_1), dim=0)
topk_indices_merged = torch.cat((topk_indices, topk_indices_1), dim=0)
#print(topk_values_merged)
#print(topk_indices_merged)

topk_values_merged_new, topk_indices_merged_new = torch.topk(topk_values_merged, k=2, dim=0)
print(topk_values_merged_new)
print(topk_indices_merged_new)
selected_indices = torch.gather(topk_indices_merged, 0, topk_indices_merged_new)
print(selected_indices)




'''
# Create a tensor to represent the batch and image indices
batch_indices = torch.arange(batch_size).unsqueeze(1).expand_as(topk_indices)

# Combine batch and image indices
image_indices = (batch_indices * batch_size + topk_indices).view(-1)

# Now image_indices is a tensor of shape (2 * num_neurons,) containing the indices of the top-2 images for each neuron
print(image_indices)
'''