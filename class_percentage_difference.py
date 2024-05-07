'''
Given for each class, the ratio of {#images in the val set}/{#images in test set}, for my split 
of the training dataset and the original split, this script calculates the absolute differences 
between the ratios between my split and the original split. 
'''

import statistics

# Given files
file_1 = f"C:\\Users\\Jasper\\Downloads\\Master thesis\\Code\\dataloaders\\imagenet_class_ratios.txt"
file_2 = f"C:\\Users\\Jasper\\Downloads\\Master thesis\\Code\\dataloaders\\imagenet_my_split_class_ratios.txt"

# File that will be created
output_txt = f"C:\\Users\\Jasper\\Downloads\\Master thesis\\Code\\dataloaders\\imagenet_class_percentage_differences.txt"

# Function to read the text file and create a dictionary of class ratios
def read_class_ratios(filename):
    class_ratios = {}
    with open(filename, 'r') as file:
        for line in file:
            # Split the line into class name and percentage
            class_name, percentage_str = line.strip().split(',')
            # Convert the percentage from string to float and strip any extra characters
            percentage = float(percentage_str.replace('%', '').strip())
            # Add to dictionary
            class_ratios[class_name] = percentage
    return class_ratios

# Read the class ratios from both files
class_ratios_1 = read_class_ratios(file_1)
class_ratios_2 = read_class_ratios(file_2)

# Calculate the absolute differences
class_differences = {}
for class_name in class_ratios_1:
    if class_name in class_ratios_2:
        diff = abs(class_ratios_1[class_name] - class_ratios_2[class_name])
        class_differences[class_name] = diff

# Write the differences to a new text file
with open(output_txt, 'w') as output_file:
    for class_name, diff in class_differences.items():
        output_file.write(f"{class_name}: {diff:.2f}%\n")

# Extract the differences for statistics calculations
differences = list(class_differences.values())

# Calculate mean, standard deviation, maximum, and minimum
mean_diff = statistics.mean(differences)
std_diff = statistics.stdev(differences)
max_diff = max(differences)
min_diff = min(differences)
median_diff = statistics.median(differences)

# Print the results
print(f"Mean of percentage differences: {mean_diff:.2f}%")
print(f"Standard deviation of percentage differences: {std_diff:.2f}%")
print(f"Maximum percentage difference: {max_diff:.2f}%")
print(f"Minimum percentage difference: {min_diff:.2f}%")
print(f"Median percentage difference: {median_diff:.2f}%")