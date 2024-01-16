import torch.nn.functional as F
import torch
import h5py

from main import original_model_output_file_path, adjusted_model_output_file_path

'''
Contents of this file:
- cross-entropy loss between original model's output and modified model's output
'''

def ce(original_model_output_file_path, adjusted_model_output_file_path):
    '''
    We don't want the model's output to change after applying the SAE. The cross-entropy is 
    suitable for comparing probability outputs. Hence, we want the cross-entropy between 
    the original model's output and the output of the modified model to be small.
    '''
    # Load the original model's output
    with h5py.File(original_model_output_file_path, 'r') as h5_file:
        original_output = torch.from_numpy(h5_file['data'][:])

    # Load the modified model's output
    with h5py.File(adjusted_model_output_file_path, 'r') as h5_file:
        adjusted_output = torch.from_numpy(h5_file['data'][:])

    # cross_entropy(input, target), where target consists of probabilities
    original_output = F.softmax(original_output, dim=1) 
    return F.cross_entropy(adjusted_output, original_output)

ce = ce(original_model_output_file_path, adjusted_model_output_file_path)
print(ce)



