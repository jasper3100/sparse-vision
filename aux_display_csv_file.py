import pandas as pd

# Replace 'your_file.csv' with the actual path to your CSV file
csv_file_path = 'evaluation_results/custom_mlp_1/cifar_10/fc1_7_0.001_64_adam_0.1_50000_sae_eval_results.csv'

# Read the CSV file into a pandas DataFrame
df = pd.read_csv(csv_file_path)

# round all entries to 2 decimal places
df = df.round(2)

# Display the DataFrame as a nice table
print(df)