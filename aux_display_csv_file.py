import pandas as pd

# Replace 'your_file.csv' with the actual path to your CSV file
csv_file_path = 'C:\\Users\\Jasper\\Downloads\\mixed3a_inceptionv1_1_0.001_512_sgd_0.1_sae_mlp_5_0.1_300_sae_eval_results.csv'
#'evaluation_results/custom_mlp_9/mnist/fc1_custom_mlp_9_1_0.1_64_sgd_0.1_sae_mlp_1_0.1_300_sae_rank_table.csv'

# Read the CSV file into a pandas DataFrame
df = pd.read_csv(csv_file_path)

# round all entries to 2 decimal places
df = df.round(2)

# sort by the "expansion_factor" column
df = df.sort_values(by='expansion_factor')

# Display the DataFrame as a nice table
print(df)