import torch

# Code to check that my computation of the polysemanticity level is correct

# set a random seed
torch.manual_seed(0)

# Assuming activations is a tensor of shape [num_samples, num_dimensions] and classes is a tensor of shape [num_samples]
c = 5  # number of classes
n = 10  # number of samples
d = 2  # number of dimensions of activation
activations = torch.randn(n, d)  
activations = torch.abs(activations)  
classes = torch.randint(0, c, (n,)) 
threshold = 1.0  

print(activations)
print(classes)

##############################
# VERSION 1: For-loop (slow)
# We create a matrix of size [d,c] where each row i, contains for a certain
# dimension i of all activations, the number of times a class j has an activation
# above the threshold.
counting_matrix = torch.zeros(d, c)
for i in range(n):
    for j in range(d):
        counting_matrix[j, classes[i]] += activations[i, j] > threshold
print(counting_matrix)

# Now, for each row i, we count the number of distinct positive integers, i.e.,
# the number of classes that have an activation above the threshold
# .bool() turns every non-zero element into a 1, and every zero into a 0
#print(counting_matrix.bool())
distinct_counts = torch.sum(counting_matrix.bool(), dim=1)
print(distinct_counts)

##############################
# VERSION 2: Vectorized (fast)

above_threshold = activations > threshold
counting_matrix = torch.zeros(d, c)
above_threshold = above_threshold.to(counting_matrix.dtype)  # Convert to the same type
counting_matrix.index_add_(1, classes, above_threshold.t())
distinct_counts = torch.sum(counting_matrix.bool(), dim=1)

print(counting_matrix)
print(distinct_counts)
# should be the same as above