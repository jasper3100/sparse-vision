import pickle
import pandas as pd

with open('machine_interpretability_dreamsim_natural_googlenet.pkl', 'rb') as f:
    data = pickle.load(f)

print(data)

scores_df = data['scores']


''' # this works
import pickle
output_file = 'data.txt'
#with open(output_file, 'w') as f:
#    f.write(str(scores_df))    
with open(output_file, 'w') as f:
    for index, row in scores_df.iterrows():
        row_str = ' '.join(map(str, row))  # Convert row to string
        f.write(row_str + '\n')
'''





#scores_df.to_excel('MIS_dreamsim_natural_googlenet_table.xlsx', index=True)

#with pd.ExcelWriter('scores_table.xlsx', engine='xlsxwriter') as writer:
#    scores_df.to_excel(writer, index=True)

'''
import pprint
with open("out.txt", "a") as f:
    pprint.pprint(data, stream=f)
'''

'''
import json
with open("out.txt", "a") as f:
    json.dump(data, f, indent=2)
'''

'''
import cPickle
output=open("write.txt","w")
output.write(str(data))
output.flush()
output.close()
'''