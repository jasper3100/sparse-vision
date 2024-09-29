import ast

'''
wnids.txt is a file with 200 lines, each line containing the word net id
of a class of Tiny Imagenet. The imagenet_label_to_wordnet_synset.txt file
contains for each Imagenet class the corresponding wordnet id. We create a file
with 200 lines, which contains for each class of Tiny Imagenet the corresponding 
index of the Imagenet class.
'''

# Define file paths
file_path_1 = 'datasets/tiny-imagenet-200/wnids.txt'
file_path_2 = 'imagenet_label_to_wordnet_synset.txt'

# Read the contents of the first file and remove the "n" from the beginning of each line
with open(file_path_2, 'r') as file:
    data = file.read()
    # Parse the data from the first file as dictionary
    parsed_data = ast.literal_eval(data)

with open(file_path_1, 'r') as file:
    # each line in wnids.txt is of the form n10348909 --> we only want to get the
    # numbers so we remove the "n"
    wnids = [line.strip()[1:] for line in file.readlines()]

# Create a dictionary to store indices
index_dict = {}

# Iterate through each wordnet ID
for wn_id in wnids:
    found = False
    # Iterate through each index and its corresponding data
    for index, value in parsed_data.items():
        # value['id] is of the orm id-n --> we want to get id
        id = value['id'].split('-')[0]
        if id == wn_id:
            index_dict[wn_id] = index
            found = True
            break
    if not found:
        index_dict[wn_id] = "error"

# Write the indices to a new file
with open('imagenet_labels_of_tiny_imagenet.txt', 'w') as file:
    for wn_id, index in index_dict.items():
        file.write(f"{index}\n")