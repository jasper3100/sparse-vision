import os
import re
import openpyxl
from openpyxl import Workbook

model_or_sae = 'model'
#model_or_sae = 'sae'

# Regular expressions for extracting the needed values
if model_or_sae == 'sae':
    sae_error_pattern = re.compile(r"Number of SAE error nodes included in circuit: (\d+)")
node_pattern = re.compile(r"(\w+)\s+(\d+) / (\d+)")
threshold_pattern = re.compile(r"Threshold values: ([\d.eE+-]+)")
faithfulness_pattern = re.compile(r"Faithfulness.*?: ([\d.eE+-]+)")

# Function to extract data from a single .out file
def extract_data(file_content):
    # Extract SAE error nodes
    if model_or_sae == 'sae':
        sae_error_nodes = int(sae_error_pattern.search(file_content).group(1))
    
    # Extract nodes left from all layers and sum them up
    nodes_left = node_pattern.findall(file_content)
    nodes_left_count = sum(int(left) for _, left, _ in nodes_left)
    
    # Add the SAE error nodes to the total node count
    if model_or_sae == 'sae':
        total_nodes_left = nodes_left_count + sae_error_nodes
    else:
        total_nodes_left = nodes_left_count
    
    # Extract threshold 1 (ignoring threshold 2)
    threshold_1 = float(threshold_pattern.search(file_content).group(1))
    
    # Extract faithfulness values and round them to 4 decimal places
    faithfulness_values = [round(float(value), 4) for value in faithfulness_pattern.findall(file_content)]
    
    # Return the extracted values
    return [total_nodes_left, threshold_1] + faithfulness_values

# Function to process all .out files and write to an Excel file
def process_files_to_excel(directory, output_excel):
    # Store all rows in a list
    rows = []
    
    # Process each .out file
    for filename in os.listdir(directory):
        if filename.endswith('.out'):
            with open(os.path.join(directory, filename), 'r') as file:
                file_content = file.read()
                row = extract_data(file_content)
                rows.append(row)
    
    # Sort the rows by threshold value (second column)
    rows.sort(key=lambda x: x[1])
    
    # Write the data to an Excel file
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Extracted Data"
    
    # Write the header
    if model_or_sae == 'sae':
        header = ['Total Nodes Left (incl. SAE Error Nodes)', 'Threshold 1', 
                'Faithfulness Zero Ablated', 'Faithfulness Mean Ablated', 'Faithfulness', 'faithfulness_deviation']
    else:
        header = ['Total Nodes Left', 'Threshold 1', 'Faithfulness', 'faithfulness_deviation']
    sheet.append(header)
    
    # Write each row of data and compute |faithfulness - 1|
    for row in rows:
        if model_or_sae == 'sae':
            # Extract the last column which contains the 'faithfulness' value
            faithfulness_value = row[4]
        else:
            # For model, the 'faithfulness' value is in the third column
            faithfulness_value = row[2]
        
        # Compute |faithfulness - 1|
        abs_faithfulness_diff = round(abs(faithfulness_value - 1), 4)
        
        # Append the computed value to the row
        row.append(abs_faithfulness_diff)
        sheet.append(row)
    
    # Save the workbook to the specified Excel file
    workbook.save(output_excel)

# Directory containing the .out files and the output CSV file
if model_or_sae == 'sae':
    name = 'sae'
else:
    name = 'model'

out_files_directory = r"C:\Users\Jasper\Downloads\Master thesis\code\evaluation_results\inceptionv1\imagenet\ie_related_quantities\out_files_" + name
output_excel_file = r"C:\Users\Jasper\Downloads\Master thesis\code\evaluation_results\inceptionv1\imagenet\ie_related_quantities\extracted_data_" + name + ".xlsx"

# Process the files and generate the CSV
#process_files_to_excel(out_files_directory, output_excel_file)

############################################################################################
# PLOT

import pandas as pd
import matplotlib.pyplot as plt

# Read the Excel file
df = pd.read_excel(output_excel_file)

# Extract the required columns
if model_or_sae == 'sae':
    x = df['Total Nodes Left (incl. SAE Error Nodes)']
else:
    x = df['Total Nodes Left']
y = df['faithfulness_deviation']

# Exclude some values 
if model_or_sae == 'sae':
    # exclude where x < 10, because they don't fit in well and are not required
    y = y[x >= 10]
    x = x[x >= 10]

# Create the scatter plot
plt.figure(figsize=(10, 6))
plt.plot(x, y, color='blue')
#plt.title('Faithfulness Deviation, ' + name)
fontsize = 22
plt.xlabel('Nodes', fontsize=fontsize, labelpad=10)
plt.ylabel('|Faithfulness - 1|', fontsize=fontsize, labelpad=10)
plt.tick_params(axis='both', which='major', labelsize=fontsize - 2)
plt.xlim(left=0)  # Ensures x-axis starts at 0
plt.ylim(bottom=0)  # Ensures y-axis starts at 0
plt.tight_layout()

# Show the plot
#plt.show()
plt.savefig(r"C:\Users\Jasper\Downloads\Master thesis\code\evaluation_results\inceptionv1\imagenet\ie_related_quantities\faithfulness_deviation_plot_" + name + ".png")