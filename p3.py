import torch

def get_top_k_samples(top_k_samples, batch_top_k_values, batch_top_k_indices, batch_filename_indices, eval_batch_idx, largest, k):
    batch_size = top_k_samples[2]

    batch_top_k_indices += (eval_batch_idx - 1) * batch_size
    # we merge the previous top k values and the current top k values --> shape: [2*k, #neurons]
    # then we find the top k values within this matrix
    top_k_values_merged = torch.cat((top_k_samples[0], batch_top_k_values), dim=0)
    # we also merge the indices and the filename indices
    top_k_indices_merged = torch.cat((top_k_samples[1], batch_top_k_indices), dim=0)
    top_k_filename_indices_merged = torch.cat((top_k_samples[3], batch_filename_indices), dim=0)
    print(top_k_samples[3], batch_filename_indices, top_k_filename_indices_merged)

    if top_k_values_merged.shape[0] < k:
        # f.e. if k = 200, but batch_size = 64, then after 2 batches, the top k values matrix has only 128 elements
        # and even after 3 batches, it has only 192 elements. Thus, we don't remove any values yet.
        top_k_values_merged_new = top_k_values_merged
        selected_indices = top_k_indices_merged
    else:
        # we find the top k values and indices of the merged top k values
        top_k_values_merged_new, top_k_indices_merged_new = torch.topk(top_k_values_merged, k=k, dim=0, largest=largest)
        # but top_k_indices_merged_new contains the indices of the top values within the top_k_values_merged
        # matrix, but we need to find the corresponding indices in top_k_indices_merged
        selected_indices = torch.gather(top_k_indices_merged, 0, top_k_indices_merged_new)
        top_k_filename_indices_merged = torch.gather(top_k_filename_indices_merged, 0, top_k_indices_merged_new)

    return (top_k_values_merged_new, selected_indices, batch_size, top_k_filename_indices_merged)

batch_top_k_values = torch.tensor([[10, 5, 4], [7, 2, 1]]) # shape: [k, #neurons] = [2,3] --> we look at top 2 values of 3 neurons
batch_top_k_indices = torch.tensor([[6, 7, 8], [9, 10, 11]])
batch_filename_indices = torch.tensor([[1,2,3], [4,5,6]]) # indices of the filenames in the batch
eval_batch_idx = 2
largest = True 
k = 2
top_k_samples = (torch.tensor([[11, 4, 3], [4, 1, 2]]), torch.tensor([[0, 1, 2], [3, 4, 5]]), 2, torch.tensor([[7,8,9],[10,11,12]]))

print(get_top_k_samples(top_k_samples, batch_top_k_values, batch_top_k_indices, batch_filename_indices, eval_batch_idx, largest, k))