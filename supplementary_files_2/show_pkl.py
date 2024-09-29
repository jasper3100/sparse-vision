# This script is used to display the content of a pickle file by
# writing it to a txt file which can be easily inspected.

import pickle
import pandas as pd

with open('machine_interpretability_dreamsim_natural_googlenet.pkl', 'rb') as f:
    data = pickle.load(f)

print(data)

scores_df = data['scores']

output_file = 'mis_reference_values.txt'

#with open(output_file, 'w') as f:
#    f.write(str(scores_df))   
with open(output_file, 'w') as f:
    for index, row in scores_df.iterrows():
        row_str = ' '.join(map(str, row))  # Convert row to string
        f.write(row_str + '\n')