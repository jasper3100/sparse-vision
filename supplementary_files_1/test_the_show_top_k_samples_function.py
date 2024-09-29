from utils import *

directory_path='C:\\Users\\Jasper\\Downloads\\Master thesis\\Code'
dataset_name='mnist'
batch_size=64
_,val_dataloader,_,_= load_data(directory_path, dataset_name, batch_size)

model_key = 'model1'
layer_name = 'layer1'

top_index_matrix = torch.tensor([[0,0,0,0],[1,1,1,1],[2,2,2,2],[3,3,3,3], [4,4,4,4], [5,5,5,5], [6,6,6,6], [7,7,7,7], [8,8,8,8], [9,9,9,9], [10,10,10,10], [11,11,11,11], [12,12,12,12], [13,13,13,13], [14,14,14,14], [15,15,15,15], [16, 16, 16, 16], [17, 17, 17, 17], [18, 18, 18, 18], [19, 19, 19, 19], [20, 20, 20, 20], [21, 21, 21, 21], [22, 22, 22, 22], [23, 23, 23, 23], [24, 24,24,24]])
# stack matrix next to itself, dim =1
top_index_matrix = torch.cat((top_index_matrix, top_index_matrix), dim=1)
top_index_matrix = torch.cat((top_index_matrix, top_index_matrix), dim=1)
top_index_matrix = torch.cat((top_index_matrix, top_index_matrix), dim=1)

top_value_matrix = top_index_matrix
# make the first column only 0s --> dead neuron
top_value_matrix[:,0] = 0	
top_k_samples = {('layer1', 'model1'): (top_value_matrix, top_index_matrix)}
small_k_samples = {('layer1', 'model1'): (top_value_matrix, top_index_matrix)}

# matrix of shape [k=n**2, num_neurons] containing the indices of the top k samples for each neuron
top_index_matrix = top_k_samples[(layer_name, model_key)][1]
small_index_matrix = small_k_samples[(layer_name, model_key)][1]
top_value_matrix = top_k_samples[(layer_name, model_key)][0]
small_value_matrix = small_k_samples[(layer_name, model_key)][0]


n = 3 # nxn top samples and nxn flop samples for each neuron
number_neurons = 4

show_top_k_samples(val_dataloader, model_key, layer_name, top_k_samples, top_k_samples,n=n, 
                   folder_path=directory_path, layer_names=None, params=None, number_neurons=number_neurons)