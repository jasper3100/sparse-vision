'''
Given the filenames of images (f.e. of the train or the val set), this script computes the classes corresponding
to those images (classes.txt) and the class counts (class_counts.txt) and stores them in a text file.

This is script is Step 1.
'''

mode = "train_train" #"train" # "eval"

# Given file:
filename_txt = f"C:\\Users\\Jasper\\Downloads\\Master thesis\\Code\\dataloaders\\imagenet_{mode}_filenames.txt"
# Files that will be created:
classes_txt = f"C:\\Users\\Jasper\\Downloads\\Master thesis\\Code\\dataloaders\\imagenet_{mode}_classes.txt"
output_file = f"C:\\Users\\Jasper\\Downloads\\Master thesis\\Code\\dataloaders\\imagenet_{mode}_class_counts.txt"

#'''
# Step 1: Extract Class IDs from imagenet_eval_filenames.txt
# Read the file and extract the class ID from each line
with open(filename_txt, 'r') as file:
    class_ids = []
    for line in file:
        # Extract the part between the first "/" and the second "/"
        # Assuming the structure "val/n02088466/ILSVRC2012_val_00038477"
        parts = line.strip().split("/")
        class_id = parts[1]  # This is the class ID
        class_ids.append(class_id)

# Step 2: Write the extracted class IDs to a new text file
with open(classes_txt, 'w') as file:
    for class_id in class_ids:
        file.write(class_id + '\n')

# Step 3: Create a mapping of class IDs to their corresponding names
class_mapping = {}
with open("C:\\Users\\Jasper\\Downloads\\Master thesis\\Code\\dataloaders\\imagenet_labels.txt", 'r') as file:
    for line in file:
        parts = line.strip().split(" ", 1)  # Split at the first space
        class_id = parts[0]
        class_name = parts[1].split(",")[0]  # Get the first class name if multiple names
        class_mapping[class_id] = class_name

# Step 4: Replace Class IDs with Class Names in imagenet_eval_classes.txt
with open(classes_txt, 'r') as file:
    class_ids = [line.strip() for line in file]

# Write back with class names instead of class IDs
with open(classes_txt, 'w') as file:
    for class_id in class_ids:
        class_name = class_mapping.get(class_id, "Unknown")  # Default to "Unknown" if not found
        file.write(class_name + '\n')
#'''

#'''
# Open the text file with class names
with open(classes_txt, 'r') as file:
    # Read all lines and remove newline characters
    class_names = [line.strip() for line in file]

# Create a dictionary to count the occurrences of each class name
class_count = {}

# Count the occurrences of each class name
for class_name in class_names:
    if class_name in class_count:
        class_count[class_name] += 1
    else:
        class_count[class_name] = 1

# Get the total number of distinct class names
distinct_class_count = len(class_count)
print(f"Total distinct class names: {distinct_class_count}")

# Sort the class names alphabetically
# We use `sorted()` on the dictionary items to sort by the key (class name)
sorted_class_counts = sorted(class_count.items(), key=lambda x: x[0])

# Write the sorted class names with counts to a new file
with open(output_file, 'w') as file:
    for class_name, count in sorted_class_counts:
        # Write the class name and its count to the file, separated by a comma
        file.write(f"{class_name}, {count}\n")
print(f"Class counts stored in '{output_file}'")
#'''
