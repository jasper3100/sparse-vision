# display the contents of a .pt file
import torch

file_path = r"C:\Users\Jasper\Downloads\Master thesis\code\evaluation_results\inceptionv1\imagenet\ie_related_quantities\IE_SAE_edges\mixed3a_inceptionv1_1_0.001_512_sgd_sae_mlp_0.001_256_constrained_adam_8_5.0_626_sae_checkpoint_epoch_7.pt"

with open(file_path, 'rb') as f:
    obj = torch.load(f)
    print(obj)