import pandas as pd

# Replace 'your_file.csv' with the actual path to your CSV file
csv_file_path = 'evaluation_results/custom_mlp_9/mnist/fc1_custom_mlp_9_1_0.1_64_sgd_0.1_sae_mlp_1_0.1_300_sae_rank_table.csv'

# Read the CSV file into a pandas DataFrame
df = pd.read_csv(csv_file_path)

# round all entries to 2 decimal places
df = df.round(2)

# Display the DataFrame as a nice table
print(df)