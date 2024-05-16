import numpy as np

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
                if len(parts) >= 3:
                    mis_value = float(parts[2])
                    total_mis += mis_value
                    count += 1
                    mis_values.append(mis_value)

    if count > 0:
        average_mis = total_mis / count
    else:
        average_mis = None
    print(count)
    median_mis = np.median(mis_values)

    return average_mis, median_mis
    
# Example usage:
file_path = "data.txt"
for layer_name in ["mixed3a", "mixed3b", "mixed4a", "mixed4b", "mixed4c", "mixed4d", "mixed4e", "mixed5a", "mixed5b"]:
    print(layer_name)
    average_mis, median_mis = extract_mis_average(file_path, layer_name)
    if average_mis is not None:
        print("Average mis value for 'mixed3a' layer:", average_mis)
    else:
        print("No rows found for 'mixed3a' layer.")
    print("Median mis value for 'mixed3a' layer:", median_mis)