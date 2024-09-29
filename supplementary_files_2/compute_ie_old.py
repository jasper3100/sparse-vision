### ORIGINAL PLAN FOR COMPUTING THE IE OF NODES BUT STORING ENCODER OUTPUTS FOR ALL SAMPLES IS 
# TOO MUCH. INSTEAD I JUST COMPUTE THE IE IN EVERY BATCH!!!

from utils import *
from get_sae_input_size import GetSaeInpSize

directory_path = 'C:\\Users\\Jasper\\Downloads\\Master thesis\\Code'
#directory_path = '/lustre/home/jtoussaint/master_thesis/'

model_name = 'inceptionv1'
dataset_name = 'imagenet'
sae_model_name = 'None'
model_weights_folder_path, sae_weights_folder_path, evaluation_results_folder_path = get_folder_paths(directory_path, model_name, dataset_name, sae_model_name)
ie_related_quantities = os.path.join(evaluation_results_folder_path, "ie_related_quantities")

model_key = "original"
layer_name = "mixed3a"
epoch = '0'
sae_checkpoint_epoch = epoch

# just for getting the right file name
model_epochs = '1'
model_learning_rate = '0.001'
batch_size = '512'
model_optimizer_name = 'sgd'

sae_model_name = 'None'
sae_epochs = '0'
sae_learning_rate = '0.0' # '0.0' #'0' #'0.0'
sae_optimizer_name = 'None'
sae_batch_size = '0'
sae_lambda_sparse = '0' #  '0.0' #'0' #'0.0'
sae_expansion_factor = '1'

dead_neurons_steps = '300'

model_params = {'model_name': model_name, 'epochs': model_epochs, 'learning_rate': model_learning_rate, 'batch_size': batch_size, 'optimizer': model_optimizer_name}
sae_params = {'sae_model_name': sae_model_name, 'sae_epochs': sae_epochs, 'learning_rate': sae_learning_rate, 'batch_size': sae_batch_size, 'optimizer': sae_optimizer_name, 'expansion_factor': sae_expansion_factor, 
                           'lambda_sparse': sae_lambda_sparse, 'dead_neurons_steps': dead_neurons_steps}

model_params_temp = {k: str(v) for k, v in model_params.items()}
sae_params_temp = {k: str(v) for k, v in sae_params.items()}
#sae_params_1_temp = {k: str(v) for k, v in sae_params_1.items()} # used for post-hoc evaluation of several models wrt expansion factor, lambda sparse, learning rate,...
params_string = '_'.join(model_params_temp.values()) + "_" + "_".join(sae_params_temp.values())
#params_string_1 = '_'.join(model_params_temp.values()) + "_" + "_".join(sae_params_1_temp.values()) # used for post-hoc evaluation

# for checkpointing, we consider all params apart from the number of epochs, because we will checkpoint at specific custom epochs
sae_params_temp.pop('sae_epochs', None)
params_string_sae_checkpoint = '_'.join(model_params_temp.values()) + "_" + "_".join(sae_params_temp.values())

device = torch.device('cpu')

used_batch_size = sae_batch_size
train_dataloader, val_dataloader, category_names, img_size = load_data(directory_path, dataset_name, used_batch_size)
num_classes = len(category_names) 

model = load_model(model_name, img_size=img_size, num_classes=num_classes)
model = model.to(device)                    
model.eval()
for param in model.parameters():
    param.requires_grad = False



def compute_ie():
    '''
    SAE FEATURE NODES

    Compute the indirect effect of SAE encoder output nodes, which correspond to a conv channel.
    Let N = total number of samples, C = number of channels, H = height, W = width, K = expansion factor of SAE
    We have: 
    - gradients of model loss wrt layer output x of shape [N, C, H, W] 
    - mean SAE encoder output on Imagenet of shape [C*K, H, W]
    - SAE encoder outputs on all circuit images of shape [N, C*K, H, W]
    - SAE decoder weight matrix of shape [C, C*K]

    The IE formula that we use assumes encoder output of dimension [C*K], i.e., for one sample out of B*H*W samples 
    i.e., we treat the convolutional layers as if they were linear layers. Hence, we do the same rearrangement/transformation
    as before: [B, C, H, W] --> [B*H*W, C]. Overall, we average the IE over B,H,W and store the IE for each layer and SAE unit i.

    SAE ERROR NODES

    For the SAE error nodes, we have:
    - gradients of model loss wrt layer output x of shape [N, C, H, W]
    - mean SAE error on Imagenet of shape [C, H, W]
    - SAE errors on all circuit images of shape [N, C, H, W]

    In this case, we take the average over all dimensions to obtain one IE value for the whole SAE error node.

    DOUBLE CHECK IF ALL DIMENSIONS ARE AS EXPECTED
    '''
    IE_sae_nodes = {}
    IE_sae_errors = {}

    for name in ["mixed3a"]: #, "mixed3b", "mixed4a", "mixed4b", "mixed4c", "mixed4d", "mixed4e", "mixed5a", "mixed5b"]:

        folder_path = os.path.join(ie_related_quantities, 'SAE_encoder_output_averages')
        file_path = get_file_path(folder_path, name, params_string_sae_checkpoint, f'epoch_{epoch}.pt')
        encoder_output_average = torch.load(file_path) # shape [C*K, H, W]
        
        folder_path = os.path.join(ie_related_quantities, 'SAE_error_averages')
        file_path = get_file_path(folder_path, name, params_string_sae_checkpoint, f'epoch_{epoch}.pt')
        sae_error_average = torch.load(file_path) # shape [C, H, W]

        folder_path = os.path.join(ie_related_quantities, 'gradients')
        file_path = get_file_path(folder_path, name, params_string_sae_checkpoint, f'epoch_{epoch}.pt')
        accumulated_gradient = torch.load(file_path) # shape [N, C, W, H]

        folder_path = os.path.join(ie_related_quantities, 'encoder_outputs')
        file_path = get_file_path(folder_path, name, params_string_sae_checkpoint, f'epoch_{epoch}.pt')
        encoder_outputs = torch.load(file_path) # shape [N, C*K, W, H]

        folder_path = os.path.join(ie_related_quantities, 'SAE_errors')
        file_path = get_file_path(folder_path, name, params_string_sae_checkpoint, f'epoch_{epoch}.pt')
        sae_errors = torch.load(file_path) # shape [N, C, W, H]

        ############## LOAD SAE TO GET SAE DECODER WEIGHT MATRIX ##############
        sae_inp_size = GetSaeInpSize(model, name, train_dataloader, device, model_name).get_sae_inp_size()
        sae_model = load_model(sae_model_name, img_size=sae_inp_size, expansion_factor=sae_expansion_factor)
        if sae_checkpoint_epoch > 0:
            file_path = get_file_path(sae_weights_folder_path, name, params=params_string_sae_checkpoint, file_name= f'sae_checkpoint_epoch_{sae_checkpoint_epoch}.pth')
            checkpoint = torch.load(file_path)
            state_dict = checkpoint['model_state_dict']
            train_batch_idx = checkpoint['training_step'] # the batch_idx counts the total number of batches used during training across epochs
            
            sae_model.load_state_dict(state_dict)
            print(f"Use SAE on layer {name} from epoch {sae_checkpoint_epoch}")
        sae_model = sae_model.to(device)

        sae_model.eval()
        for param in sae_model.parameters():
            param.requires_grad = False

        decoder_weight_matrix = sae_model.decoder.weight.data # shape [C, C*K] where K is the expansion factor
        # the feature direction (unit vector) v_i is the i-th column of the decoder weight matrix
        #####################################################################

        N = accumulated_gradient.shape[0]

        print(name, encoder_outputs.shape, sae_errors.shape, encoder_output_average.shape, sae_error_average.shape, accumulated_gradient.shape)

        encoder_output_average = torch.unsqueeze(encoder_output_average, 0) # add a new dimension at the beginning -> shape: [1, C*K, H, W]
        encoder_output_average = encoder_output_average.repeat(N, 1, 1, 1) # repeat the tensor N times along the new dimension -> shape: [N, C*K, H, W]
        encoder_output_average = rearrange(encoder_output_average, 'b c h w -> (b h w) c') # unfold the tensor to shape [NHW, C*K]

        sae_error_average = torch.unsqueeze(sae_error_average, 0) # add a new dimension at the beginning -> shape: [1, C, H, W]
        sae_error_average = sae_error_average.repeat(N, 1, 1, 1) # repeat the tensor N times along the new dimension -> shape: [N, C, H, W]
        sae_error_average = rearrange(sae_error_average, 'b c h w -> (b h w) c') # unfold the tensor to shape [NHW, C]
        
        encoder_outputs = rearrange(encoder_outputs, 'b c h w -> (b h w) c') # shape [NHW, C*K]
        accumulated_gradient = rearrange(accumulated_gradient, 'b c h w -> (b h w) c') # shape [NHW, C], 
        # recall that these are the gradients with respect to the layer output in the original model, hence we have C channels
        sae_errors = rearrange(sae_errors, 'b c h w -> (b h w) c') # shape [NHW, C]

        CxK = encoder_output_average.shape[1] # C*K

        ############## COMPUTE IE FOR EACH SAE NODE / CHANNEL ##############
        for channel_idx in range(CxK):
            feature_direction_i = decoder_weight_matrix[:, channel_idx] # v_i, shape [C, 1]
            encoder_output_i = encoder_outputs[:, channel_idx] # shape [NHW, 1]
            encoder_output_average_i = encoder_output_average[:, channel_idx] # shape [NHW, 1]

            # by the chain rule
            grad = accumulated_gradient @ feature_direction_i # matrix multiplication: [NHW, C] @ [C, 1] = [NHW, 1]

            '''
            In the basic case (IE formula (3) presented in Marks paper), without conv layer (H=W=1) and one sample (N=1), we have [1, C] @ [C, 1] = [1],
            where C is the number of units. Now, for N samples, i.e., if we are given tensors of shape [N, C] and [C, N], we want to get 10 scalar values
            by multiplying each row of the first tensor by the corresponding column of the second tensor. These 10 values correspond to the 10 values that
            one obtains by doing matrix multiplication of those tensors and then considering the diagonal elements (which are the result of multipling the 
            i-th row by the i-th column). Hence, we do:
            [NHW, C] @ [NHW, C].T = [NHW, C] @ [C, NHW] = [NHW, NHW] and then take the diagonal elements --> [NHW]. Then, we take the mean.

            For the SAE error nodes we have C = number of channels in corresponding layer of original model. However, for the SAE feature nodes, we have C=1. 
            In this case, the procedure can be simplified (even though I just use the same procedure as for the SAE error nodes, because it is more general).
            If C=1, we don't need to worry about taking the dot product of two vectors, but instead, for each N,H,W we can simply do scalar multiplication of
            gradient and (encoder_output_average_i - encoder_output_i). Doing this for all N,H,W, we can just do pointwise multiplication: 
            [NHW, 1] * [NHW, 1] = [NHW, 1], and then we can take the mean over the first dimension.

            Code: 
            IE = grad * (encoder_output_average_i - encoder_output_i)
            IE = torch.mean(IE, dim=0).item() # shape: scalar

            Example code for demonstration:
            A = torch.rand(10,1)
            B = torch.rand(1,10)

            print(torch.diag(A @ B).shape)
            print((A * B.T).shape)
            '''
            IE = torch.diag(grad @ (encoder_output_average_i - encoder_output_i).T) # shape: [NHW]
            # we take the mean over NHW, i.e., over all "samples"
            IE_sae_nodes[(name,channel_idx)] = torch.mean(IE) # shape: scalar

        ############## COMPUTE IE FOR SAE ERROR ##############
        IE = torch.diag(accumulated_gradient @ (sae_error_average - sae_errors).T) # shape: [NHW]
        IE_sae_errors[name] = torch.mean(IE) # shape: scalar

    
    # store the IE values
    file_path = get_file_path(ie_related_quantities, None, params_string_sae_checkpoint, f'epoch_{epoch}_ie_sae_nodes.pt')
    torch.save(IE_sae_nodes, file_path)
    file_path = get_file_path(ie_related_quantities, None, params_string_sae_checkpoint, f'epoch_{epoch}_ie_sae_errors.pt')
    torch.save(IE_sae_errors, file_path)