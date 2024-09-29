import os
from tiny_imagenet import TinyImageNetPaths

root_dir='datasets/tiny-imagenet-200'
batch_size=32
# if root_dir does not exist, download the dataset
download = not os.path.exists(root_dir)

# Example usage
root_dir = 'datasets/tiny-imagenet-200'
tiny_imagenet_paths = TinyImageNetPaths(root_dir, download=False)

# Get a list of all category names
all_category_names = []
for class_id, category_names in tiny_imagenet_paths.nid_to_words.items():
    all_category_names.extend(category_names)

# Print the list of all category names
print("List of all category names:")
print(all_category_names)