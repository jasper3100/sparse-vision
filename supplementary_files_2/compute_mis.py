import numpy as np

'''
Given a file of per unit MIS values. This script extracts the average and median MIS value for a given layer.
'''

def extract_mis_average(file_path, layer_name):
    # Initialize variables
    total_mis = 0
    count = 0

    # Open the file
    with open(file_path, 'r') as file:
        # Read each line
        mis_values = []
        for line in file:
            # Split the line into parts
            parts = line.split()
            # Check if layer_name is "mixed3a"
            if len(parts) >= 1 and layer_name in parts[0] and "bottleneck" not in parts[0]:
                # Extract mis value
                if len(parts) >= 4:
                    mis_value = float(parts[3]) # mis value is in the column "mis_confidence"
                    total_mis += mis_value
                    count += 1
                    mis_values.append(mis_value)
                else:
                    print("weird line")

    if count > 0:
        average_mis = total_mis / count
    else:
        average_mis = None
    print(count)
    median_mis = np.median(mis_values)

    return average_mis, median_mis
    
# given a csv file convert it to a txt file
def convert_csv_to_txt(file_path):
    # Open the file
    with open(file_path, 'r') as file:
        # Read each line
        with open("mis_values.txt", 'w') as output_file:
            for line in file:
                # Split the line into parts
                parts = line.split(",")
                # Reorder the parts
                parts = [parts[4], parts[0], parts[2], parts[3]] # layer_name, idx, mis, mis_confidence
                if parts[0] != "layer_name": # we don't keep the first line of column names
                    # Write the line to the output file and add a new line
                    output_file.write(" ".join(parts) + "\n")
    
#file_path = 'C:\\Users\\Jasper\\Downloads\\Master thesis\\Code\\evaluation_results\\inceptionv1\\imagenet\\MIS\\original_mixed3a_inceptionv1_1_0.001_512_sgd_None_0_0.0_0_None_1_0_300_mis_epoch_0.csv'

#convert_csv_to_txt(file_path)

# Example usage:
file_path = "mis_reference_values.txt" # "mis_values.txt"
#'''
for layer_name in ["mixed3a", "mixed3b", "mixed4a", "mixed4b", "mixed4c", "mixed4d", "mixed4e", "mixed5a", "mixed5b"]:
    print(layer_name)
    average_mis, median_mis = extract_mis_average(file_path, layer_name)
    if average_mis is not None:
        print(f"Average mis value for {layer_name} layer:", average_mis)
    else:
        print(f"No rows found for {layer_name} layer.")
    print(f"Median mis value for {layer_name} layer:", median_mis)
#'''
