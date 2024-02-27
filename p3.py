import torch
import torch.nn.functional as F

# Example data (replace with your actual data)
neuron_activity = torch.tensor([
    [1, 2, 3, 4, 5],  # Neuron 1
    [0, 1, 0, 2, 0],  # Neuron 2
    [1, 1, 1, 1, 1],  # Neuron 3
    [1,0,0,0,0],
    [0,0,0,0,1],
    [100,0,0,0,0],
    # ... more neurons
], dtype=torch.float32)

# Convert neuron activity to probabilities (softmax)
probabilities = F.softmax(neuron_activity, dim=1)

print(probabilities)

# Calculate Shannon entropy for each neuron
entropy_scores = -torch.sum(probabilities * torch.log2(probabilities + 1e-8), dim=1)

# Normalize scores between 0 and 1 (optional)
normalized_scores = entropy_scores / torch.log2(torch.tensor(neuron_activity.size(1), dtype=torch.float32))

# Convert to Python floats for display
entropy_scores = entropy_scores.tolist()
normalized_scores = normalized_scores.tolist()


# Replace the conversion to Python lists with this
entropy_scores = [round(float(score),3) for score in entropy_scores]
normalized_scores = [round(float(score),3) for score in normalized_scores]

# Display the results
print("Neuron Activity:")
print(neuron_activity)
print("\nShannon Entropy Scores:")
print(entropy_scores)
print("\nNormalized Scores:")
print(normalized_scores)