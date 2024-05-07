'''
Once the the class counts of the training and evaluation sets were computed using "class_counts.py", the following code
calculates the proportion of the number of images of a certain class compared between train and val set.
''' 

import statistics

'''
mode1 = "train_train"
mode2 = "train_val"
name = "imagenet_my_split_class_ratios"
'''

#'''
mode1 = "train"
mode2 = "eval" 
name = "imagenet_class_ratios"
#'''

# Given files
output_file_1 = f"C:\\Users\\Jasper\\Downloads\\Master thesis\\Code\\dataloaders\\imagenet_{mode1}_class_counts.txt"
output_file_2 = f"C:\\Users\\Jasper\\Downloads\\Master thesis\\Code\\dataloaders\\imagenet_{mode2}_class_counts.txt"
# File that will be created
class_ratios_file = f"C:\\Users\\Jasper\\Downloads\\Master thesis\\Code\\dataloaders\\{name}.txt"

# Step 1: Read the class counts from the first file
class_counts_1 = {}
with open(output_file_1, 'r') as file:
    for line in file:
        class_name, count = line.strip().split(", ")
        class_counts_1[class_name] = int(count)

# Step 2: Read the class counts from the second file
class_counts_2 = {}
with open(output_file_2, 'r') as file:
    for line in file:
        class_name, count = line.strip().split(", ")
        class_counts_2[class_name] = int(count)

# Step 3: Calculate the percentage of smaller to larger class count
# Combine all class names from both dictionaries
all_class_names = set(class_counts_1.keys()) | set(class_counts_2.keys())

# Create a new dictionary to store the calculated percentages
class_ratios = {}

for class_name in all_class_names:
    # Get the class count from both dictionaries, defaulting to 0 if not found
    count1 = class_counts_1.get(class_name, 0)
    count2 = class_counts_2.get(class_name, 0)

    # Determine the smaller and larger counts
    small_count = min(count1, count2)
    large_count = max(count1, count2)

    # Calculate the percentage (avoid division by zero)
    if large_count > 0:
        percentage = small_count / large_count * 100
    else:
        percentage = 0

    class_ratios[class_name] = percentage

# Step 4: Write the calculated percentages to a new text file
with open(class_ratios_file, 'w') as file:
    for class_name, percentage in sorted(class_ratios.items()):
        # Write the class name and the percentage to the file
        file.write(f"{class_name}, {percentage:.2f}%\n")
        #if percentage < 3.4:
        #    print(f"{class_name}, {percentage:.2f}%")
#'''

#'''
# Extract the values (percentages) from the dictionary
percentages = list(class_ratios.values())

# Calculate the range (difference between max and min)
range_percentages = max(percentages) - min(percentages)

# Calculate the mean & std
mean_percentages = statistics.mean(percentages)
std_percentages = statistics.stdev(percentages)
median_percentages = statistics.median(percentages)

print(max(percentages))
print(min(percentages))
print("Range of percentages:", range_percentages)
print("Mean of percentages:", mean_percentages)
print("Standard deviation of percentages:", std_percentages)
print("Median of percentages:", median_percentages)
#'''