import pandas as pd

# Replace 'your_file.csv' with the actual path to your CSV file
#csv_file_path = 'C:\\Users\\Jasper\\Downloads\\mixed3a_inceptionv1_1_0.001_512_sgd_0.1_sae_mlp_5_0.1_300_sae_eval_results.csv'
csv_file_path = r"C:\Users\Jasper\Downloads\Master thesis\code\evaluation_results\inceptionv1\imagenet\mixed3a_inceptionv1_1_0.001_512_sgd_sae_mlp_199_sae_eval_results.csv"
#'evaluation_results/custom_mlp_9/mnist/fc1_custom_mlp_9_1_0.1_64_sgd_0.1_sae_mlp_1_0.1_300_sae_rank_table.csv'

# Read the CSV file into a pandas DataFrame
df = pd.read_csv(csv_file_path)

# round all entries to 2 decimal places
df = df.round(2)

# sort by the "expansion_factor" column
df = df.sort_values(by='expansion_factor')

# remove rows with expansion factor 0.0, 3.0, approximately equal to 0.05
#df = df[df['expansion_factor']!=0.0]
#df = df[df['expansion_factor']!=3.0]
#df = df[df['expansion_factor'].round(2)!=0.05]

print(df['expansion_factor'].unique())
print(df['lambda_sparse'].unique())

# overwrite the original CSV file
#df.to_csv(csv_file_path, index=False)

# Display the DataFrame as a nice table

# dont include the optimizer_name and the median_mis column
df = df.drop(columns=['optimizer_name', 'median_mis'])
#print(df)

# show only columns with lambda = 5.0
df = df[df['lambda_sparse']==5.0]
# show only the column rel_sparsity
df = df[['epochs','rel_sparsity']]
print(df)
# store the sparsity values in a list
data = list(zip(df['epochs'], df['rel_sparsity']))
print(data)