import torch

# Example data
number_active_classes_per_neuron = torch.rand((5, 3))  # Replace with your actual matrix
dead_neurons = torch.tensor([True, False, True, False, True])  # Replace with your actual tensor

if len(dead_neurons) != number_active_classes_per_neuron.shape[0]:
    raise ValueError("The length of dead_neurons should be equal to the number of rows in number_active_classes_per_neuron")

# Remove rows where self.dead_neurons[name] is False
filtered_matrix = number_active_classes_per_neuron[dead_neurons == False]


# Display the result
print("Original Matrix:")
print(number_active_classes_per_neuron)
print("\nFiltered Matrix:")
print(filtered_matrix)
