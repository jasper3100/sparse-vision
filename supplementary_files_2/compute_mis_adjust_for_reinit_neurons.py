# Take runs with ids 195 and 196 and remove freshly re-initialized neurons from MIS values

import pandas as pd
import matplotlib.pyplot as plt
 
sae_eval_results_195 = r"C:\Users\Jasper\Downloads\Master thesis\code\evaluation_results\inceptionv1\imagenet\mixed3a_inceptionv1_1_0.001_512_sgd_sae_mlp_195_sae_eval_results.csv"
sae_eval_results_195 = pd.read_csv(sae_eval_results_195)
sae_eval_results_196 = r"C:\Users\Jasper\Downloads\Master thesis\code\evaluation_results\inceptionv1\imagenet\mixed3a_inceptionv1_1_0.001_512_sgd_sae_mlp_196_sae_eval_results.csv"
sae_eval_results_196 = pd.read_csv(sae_eval_results_196)

sae_eval_results_625 = r"C:\Users\Jasper\Downloads\Master thesis\code\evaluation_results\inceptionv1\imagenet\mixed3a_inceptionv1_1_0.001_512_sgd_sae_mlp_625_sae_eval_results.csv"
sae_eval_results_625 = pd.read_csv(sae_eval_results_625)
sae_eval_results_626 = r"C:\Users\Jasper\Downloads\Master thesis\code\evaluation_results\inceptionv1\imagenet\mixed3a_inceptionv1_1_0.001_512_sgd_sae_mlp_626_sae_eval_results.csv"
sae_eval_results_626 = pd.read_csv(sae_eval_results_626)

# only keep those rows where mis != 0
sae_eval_results_625 = sae_eval_results_625[sae_eval_results_625['median_mis'] != 0]
sae_eval_results_626 = sae_eval_results_626[sae_eval_results_626['median_mis'] != 0]
# sort by epochs
sae_eval_results_625 = sae_eval_results_625.sort_values(by='epochs')
sae_eval_results_626 = sae_eval_results_626.sort_values(by='epochs')

# only keep those rows where mis != 0
sae_eval_results_195 = sae_eval_results_195[sae_eval_results_195['median_mis'] != 0]
sae_eval_results_196 = sae_eval_results_196[sae_eval_results_196['median_mis'] != 0]
# sort by epochs
sae_eval_results_195 = sae_eval_results_195.sort_values(by='epochs')
sae_eval_results_196 = sae_eval_results_196.sort_values(by='epochs')

# MIS values of baseline 
# List of points for medians
medians_1 = [
    (1, 0.854),
    (2, 0.858),
    (3, 0.863),
    (4, 0.865),
    (5, 0.867),
    (6, 0.867),
    (7, 0.864),
    # Skipping epoch 8 as mentioned
    (9, 0.852),
    #(10, 0.498),
    (11, 0.837),
    (12, 0.833),
    (13, 0.825),
    (14, 0.824),
    (15, 0.822)
]

# we dont adjust anything for 626 because there I re-init. neurons weirdly, so that theyre not re-init right before the end of an epoch... but rather at step 7, 1259, 2511, 3763 within one epoch... (one epoch has 5002 steps...)

for id in [195, 196, 199, 625]:
    if id == 199:
        epochs = [2,4,6,9] # only dead neuron indices for those epochs exist, also at epoch 8 but at epoch 8 no MIS values
        mlp_epochs = 10
    elif id == 195:
        epochs = [4,6,8]
        mlp_epochs = 16
    elif id == 196:
        epochs = [4,5,7,8,9,10] # somehow epoch 6 is missing
        mlp_epochs = 16
    elif id == 625:
        epochs = [1,3,4,5,7,8,9,10]
        mlp_epochs = 11

    for epoch in epochs:
        mis_values_file = fr"C:\Users\Jasper\Downloads\Master thesis\code\evaluation_results\inceptionv1\imagenet\MIS\sae_mixed3a_inceptionv1_1_0.001_512_sgd_sae_mlp_{mlp_epochs}_0.001_256_constrained_adam_8_5.0_{id}_mis_epoch_{epoch}.csv"

        if id == 195 or id == 196 or id == 625 or id == 626:
            if id == 195 or id == 196:
                batch_idx_total = 20007 + (epoch - 4) * 5002
                batch_idx_in_epoch = 5001
            elif id == 625:
                batch_idx_total = epoch * 5000 + 1
                batch_idx_in_epoch = 5001 - (epoch - 1) * 2
            # indices of reinitialized neurons
            # we should not take indices of dead neurons because they are measured
            # between step 5001 and 5002, so it could be many neurons but not
            # representative
            indices_file = fr"C:\Users\Jasper\Downloads\Master thesis\code\evaluation_results\inceptionv1\imagenet\indices_of_re_initialized_neurons\mixed3a_inceptionv1_1_0.001_512_sgd_sae_mlp_0.001_256_constrained_adam_8_5.0_{id}_epoch_{epoch}_train_batch_idx_{batch_idx_total}_epoch_batch_idx_{batch_idx_in_epoch}.txt"
        elif id == 199:
            # indices of dead neurons
            indices_file = fr"C:\Users\Jasper\Downloads\Master thesis\code\evaluation_results\inceptionv1\imagenet\indices_of_dead_neurons\mixed3a_inceptionv1_1_0.001_512_sgd_sae_mlp_0.001_256_constrained_adam_8_5.0_199_epoch_{epoch}_epoch_batch_idx_5002.txt"

        mis_df = pd.read_csv(mis_values_file)
        with open(indices_file, 'r') as file:
            indices = file.read().splitlines()

        # Convert the indices to integers if they are in string format
        indices = list(map(int, indices))

        # drop the rows at the indices
        mis_df = mis_df.drop(indices)

        # get the MIS values which are in the column "MIS_confidence"
        mis_values = mis_df["MIS_confidence"]

        # compute median 
        median = mis_values.median()
        
        # go to the row in sae_eval_results_195/196 where the epoch is equal to the current epoch and replace the median_mis value with the computed median
        if id == 195:
            sae_eval_results_195.loc[sae_eval_results_195['epochs'] == epoch, 'median_mis'] = median
        elif id == 196:
            sae_eval_results_196.loc[sae_eval_results_196['epochs'] == epoch, 'median_mis'] = median
        elif id == 625:
            sae_eval_results_625.loc[sae_eval_results_625['epochs'] == epoch, 'median_mis'] = median
        elif id == 199:
            # cannot do epoch-1 here because index 8 is missing f.e. in medians_1
            medians_1 = [(epoch, median) if t[0] == epoch else t for t in medians_1]
                
# plot
plt.plot(*zip(*medians_1), color="gold", label='No re-init.')
plt.plot(sae_eval_results_195['epochs'], sae_eval_results_195['median_mis'],  color="tab:red", label='(1) Re-init. at epochs 4,6,8')
plt.plot(sae_eval_results_196['epochs'], sae_eval_results_196['median_mis'], color="tab:green", label='(2) Re-init. at epochs 3-10')
plt.plot(sae_eval_results_625['epochs'], sae_eval_results_625['median_mis'], color="tab:orange", label='(3) Re-init. at all epochs')
plt.plot(sae_eval_results_626['epochs'], sae_eval_results_626['median_mis'], color="tab:blue", label='(4) Re-init. at all epochs with rescaling')
plt.xlabel('Epoch')
plt.ylabel('Median MIS')
plt.ylim(0.85, 0.875)
plt.xlim(1,10)
#plt.title('Median MIS of SAE on mixed3 with exp fac 8')
#plt.axhline(y=0.834, color='tab:gray', linestyle='--', label='Model layer')
#plt.legend()
plt.gca().spines['top'].set_color('none')
plt.gca().spines['right'].set_color('none')
plt.gca().spines['left'].set_color('darkgrey')
plt.gca().spines['bottom'].set_color('darkgrey')
plt.gca().xaxis.set_tick_params(color='darkgrey')
plt.gca().yaxis.set_tick_params(color='darkgrey')
plt.gca().xaxis.label.set_color('dimgrey')
plt.gca().yaxis.label.set_color('dimgrey')
plt.setp(plt.gca().get_xticklabels(), color='dimgrey')
plt.setp(plt.gca().get_yticklabels(), color='dimgrey')
plt.gca().xaxis.set_label_coords(0.95, 0.05)
plt.gca().yaxis.set_label_coords(0.05, 0.9)
#plt.show()
plt.savefig(r"C:\Users\Jasper\Downloads\Master thesis\code\evaluation_results\inceptionv1\imagenet\mixed3a_inceptionv1_1_0.001_512_sgd_sae_mlp_195_196_199_median_mis_adjusted_for_reinit_and_dead_neurons.png", dpi=300)