import torch
import torch.nn.functional as F

# each row represents one neuron
# each column represents one class
# --> (i,j)-th entry is how often neuron i was active for class j
neuron_activity = torch.tensor([
    [1, 2, 3, 4, 5],  
    [0, 1, 0, 2, 0],  
    [1, 1, 1, 1, 1],  
    [1,0,0,0,0],
    [0,0,0,0,1],
    [100,0,0,0,0],
    [100,0,0,0,110],
], dtype=torch.float32)

# Convert neuron activity to probabilities (softmax)
probabilities = F.softmax(neuron_activity, dim=1)

# Calculate Shannon entropy for each neuron
entropy_scores = -torch.sum(probabilities * torch.log2(probabilities + 1e-8), dim=1)


# normalize the neuron activity
#neuron_activity = neuron_activity / neuron_activity.sum(dim=1, keepdim=True)
#print(neuron_activity)

# Normalize scores between 0 and 1 (optional)
normalized_scores = entropy_scores / torch.log2(torch.tensor(neuron_activity.size(1), dtype=torch.float32))
normalized_scores = normalized_scores.tolist()
normalized_scores = [round(float(score),3) for score in normalized_scores]


# Convert to Python floats for display
entropy_scores = entropy_scores.tolist()

# Replace the conversion to Python lists with this
entropy_scores = [round(float(score),3) for score in entropy_scores]

# Display the results
print("Neuron Activity:")
print(neuron_activity)
print("\nShannon Entropy Scores:")
print(entropy_scores)
print("\nNormalized Scores:")
print(normalized_scores)

gini_scores = 1 - torch.sum(probabilities**2, dim=1)
gini_scores = gini_scores.tolist()
gini_scores = [round(float(score),3) for score in gini_scores]

# Calculate mean and standard deviation for each neuron
mean_activity = torch.mean(probabilities, dim=1)
std_activity = torch.std(probabilities, dim=1)

# calculate the variance
var_activity = torch.var(neuron_activity, dim=1)
var_activity = var_activity.tolist()
var_activity = [round(float(score),3) for score in var_activity]

print("\nVariance:")
print(var_activity)

'''
# Calculate coefficient of variation for each neuron
cv_scores = std_activity / mean_activity
cv_scores = cv_scores.tolist()
cv_scores = [round(float(score),3) for score in cv_scores]

print("\nGini Scores:")
print(gini_scores)
print("\nCoefficient of Variation Scores:")
print(cv_scores)
'''