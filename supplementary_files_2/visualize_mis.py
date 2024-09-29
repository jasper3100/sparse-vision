import matplotlib.pyplot as plt
import pandas as pd

########################## MIXED3A EXP FAC 8 LAMBDA 5 ##########################
'''
dead_neurons_steps = 199
- epoch 1: median 0.854, average 0.847; m 0.854, a 0.848
- epoch 2: m: 0.858, a 0.854
- epoch 3: m 0.863, a 0.858; m 0.862, a 0.857
- epoch 4: m 0.865, a 0.860
- epoch 5: m 0.867, a 0.861, m 0.867, a 0.861
- epoch 6: m 0.867, a 0.861
- epoch 7: m 0.864, a 0.856; m 0.863, a 0.855
- epoch 8 didn’t terminate for some reason, didn’t start again because the down trend is clearly visible here
- epoch 9: m 0.852, a 0.846
- epoch 10: m 0.498, a 0.498; m 0.499, a 0.499
- epoch 11: m 0.837, a 0.833
- epoch 12: m 0.833, a 0.828
- epoch 13: m 0.825, a 0.821
- epoch 14: m 0.824, a 0.820
- epoch 15: m 0.822, a 0.818

The MIS of the GoogLeNet layer 3a was: median: 0.834, average: 0.804
'''

lam = 5
exp_fac = 8

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
    #(10, 0.498), # weird point
    (11, 0.837),
    (12, 0.833),
    (13, 0.825),
    (14, 0.824),
    (15, 0.822)
]

# List of points for averages
averages_1 = [
    (1, 0.847),
    (2, 0.854),
    (3, 0.858),
    (4, 0.860),
    (5, 0.861),
    (6, 0.861),
    (7, 0.856),
    # Skipping epoch 8 as mentioned
    (9, 0.846),
    (10, 0.498),
    (11, 0.833),
    (12, 0.828),
    (13, 0.821),
    (14, 0.820),
    (15, 0.818)
]

########################## MIXED3A EXP FAC 8 LAMBDA 0 ##########################

# dead_neurons_steps = 199
lam = 0
exp_fac = 8

medians_2 = [
    # 1 didn not terminate for some reason
    (2, 0.857),
    (3, 0.861),
    (4, 0.864),
    (5, 0.865),
    (6, 0.866),
    (7, 0.863),
    (8, 0.858),
    (9, 0.852),
    (10, 0.843),
    (11, 0.839),
    (12, 0.832),
    (13, 0.831),
    (14, 0.822),
    (15, 0.815),
]

averages_2 = [
    # 1 didn't terminate for some reason
    (2, 0.853),
    (3, 0.857),
    (4, 0.859),
    (5, 0.860),
    (6, 0.860),
    (7, 0.857),
    (8, 0.853),
    (9, 0.847),
    (10, 0.838),
    (11, 0.834),
    (12, 0.828),
    (13, 0.827),
    (14, 0.820),
    (15, 0.813),
]



'''
plt.plot(*zip(*medians_2), label='Median')
plt.plot(*zip(*averages_2), label='Average')
plt.xlabel('Epoch')
plt.ylabel('MIS')
plt.title(f"MIS of SAE on mixed3a with exp fac {str(exp_fac)} and lambda {str(lam)}")
plt.ylim(0.8, 0.88)
plt.axhline(y=0.834, color='b', linestyle='--', label='Original layer median MIS')
plt.axhline(y=0.804, color='orange', linestyle='--', label='Original layer average MIS')
plt.legend()
plt.show()
'''

'''
plt.plot(*zip(*medians_1), label='Median lambda 5')
plt.plot(*zip(*medians_2), label='Median lambda 0')
plt.xlabel('Epoch')
plt.ylabel('MIS')
plt.title(f"MIS of SAE on mixed3a with exp fac {str(exp_fac)}")
plt.ylim(0.82, 0.87)
plt.axhline(y=0.834, color='b', linestyle='--', label='Original layer median MIS')
plt.legend()
plt.show()
'''

########################## MIXED3A EXP FAC 8 INCREASING LAMBDA ##########################

# dead_neuron_steps = 194
#sae_eval_results_194 = r"C:\Users\Jasper\Downloads\Master thesis\code\evaluation_results\inceptionv1\imagenet\mixed3a_inceptionv1_1_0.001_512_sgd_sae_mlp_194_sae_eval_results.csv"
#sae_eval_results_194 = pd.read_csv(sae_eval_results_194)
#sae_eval_results_193 = r"C:\Users\Jasper\Downloads\Master thesis\code\evaluation_results\inceptionv1\imagenet\mixed3a_inceptionv1_1_0.001_512_sgd_sae_mlp_193_sae_eval_results.csv"
#sae_eval_results_193 = pd.read_csv(sae_eval_results_193)

'''
# only keep those rows where mis != 0 
sae_eval_results_194 = sae_eval_results_194[sae_eval_results_194['median_mis'] != 0]
sae_eval_results_193 = sae_eval_results_193[sae_eval_results_193['median_mis'] != 0]
# sort by epochs
sae_eval_results_194 = sae_eval_results_194.sort_values(by='epochs')
sae_eval_results_193 = sae_eval_results_193.sort_values(by='epochs')
'''

'''
# plot
plt.plot(sae_eval_results_194['epochs'], sae_eval_results_194['median_mis'], label='Increasing lambda run 1')
plt.plot(sae_eval_results_193['epochs'], sae_eval_results_193['median_mis'], label='Increasing lambda run 2')
plt.plot(*zip(*medians_1), label='Median lambda 5')
plt.xlabel('Epoch')
plt.ylabel('Median MIS')
plt.legend()
plt.ylim(0.82, 0.88)
plt.xlim(0,10)
plt.title('Median MIS of SAE on mixed3 with exp fac 8')
plt.axhline(y=0.834, color='b', linestyle='--', label='Original layer median MIS')
plt.show()
'''
            
# more beautiful plot (in order to be similar to W&B plots)
'''
plt.plot(*zip(*medians_1), color="gold", label='No re-init.')
plt.plot(sae_eval_results_193['epochs'], sae_eval_results_193['median_mis'],  color="tab:red", label='Increasing $\lambda$')
#plt.plot(sae_eval_results_194['epochs'], sae_eval_results_194['median_mis'], color="tab:green", label='Increasing1')
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
plt.show()
'''
#plt.savefig(r"C:\Users\Jasper\Downloads\abc.png", dpi=300)


########################## MIXED3A EXP FAC 8 RE-INITIALIZATIONS ##########################

#sae_eval_results_195 = r"C:\Users\Jasper\Downloads\Master thesis\code\evaluation_results\inceptionv1\imagenet\mixed3a_inceptionv1_1_0.001_512_sgd_sae_mlp_195_sae_eval_results.csv"
#sae_eval_results_195 = pd.read_csv(sae_eval_results_195)
#sae_eval_results_196 = r"C:\Users\Jasper\Downloads\Master thesis\code\evaluation_results\inceptionv1\imagenet\mixed3a_inceptionv1_1_0.001_512_sgd_sae_mlp_196_sae_eval_results.csv"
#sae_eval_results_196 = pd.read_csv(sae_eval_results_196)

'''
# only keep those rows where mis != 0
sae_eval_results_195 = sae_eval_results_195[sae_eval_results_195['median_mis'] != 0]
sae_eval_results_196 = sae_eval_results_196[sae_eval_results_196['median_mis'] != 0]
# sort by epochs
sae_eval_results_195 = sae_eval_results_195.sort_values(by='epochs')
sae_eval_results_196 = sae_eval_results_196.sort_values(by='epochs')

# plot
plt.plot(sae_eval_results_195['epochs'], sae_eval_results_195['median_mis'], label='Re-initialization at epochs 4,6,8')
plt.plot(sae_eval_results_196['epochs'], sae_eval_results_196['median_mis'], label='Re-initialization at epochs 3-10')
plt.plot(*zip(*medians_1), label='Median lambda 5')
plt.xlabel('Epoch')
plt.ylabel('Median MIS')
plt.legend()
plt.ylim(0.82, 0.88)
plt.xlim(0,10)
plt.title('Median MIS of SAE on mixed3 with exp fac 8')
plt.axhline(y=0.834, color='b', linestyle='--', label='Original layer median MIS')
plt.show()
'''

########################## MIXED3A EXP FAC 8 RE-INITIALIZATIONS V2 ##########################

#sae_eval_results_625 = r"C:\Users\Jasper\Downloads\Master thesis\code\evaluation_results\inceptionv1\imagenet\mixed3a_inceptionv1_1_0.001_512_sgd_sae_mlp_625_sae_eval_results.csv"
#sae_eval_results_625 = pd.read_csv(sae_eval_results_625)
#sae_eval_results_626 = r"C:\Users\Jasper\Downloads\Master thesis\code\evaluation_results\inceptionv1\imagenet\mixed3a_inceptionv1_1_0.001_512_sgd_sae_mlp_626_sae_eval_results.csv"
#sae_eval_results_626 = pd.read_csv(sae_eval_results_626)

'''
# only keep those rows where mis != 0
sae_eval_results_625 = sae_eval_results_625[sae_eval_results_625['median_mis'] != 0]
sae_eval_results_626 = sae_eval_results_626[sae_eval_results_626['median_mis'] != 0]
# sort by epochs
sae_eval_results_625 = sae_eval_results_625.sort_values(by='epochs')
sae_eval_results_626 = sae_eval_results_626.sort_values(by='epochs')

# plot
plt.plot(sae_eval_results_625['epochs'], sae_eval_results_625['median_mis'], label='Re-initialization every quater epoch with rescaling')
plt.plot(sae_eval_results_626['epochs'], sae_eval_results_626['median_mis'], label='Re-initialization every quarter epoch')
plt.plot(*zip(*medians_1), label='Without re-initializations')
plt.xlabel('Epoch')
plt.ylabel('Median MIS')
plt.legend()
plt.ylim(0.82, 0.88)
plt.xlim(0,10)
plt.title('Median MIS of SAE on mixed3a with exp fac 8')
plt.axhline(y=0.834, color='b', linestyle='--', label='Original layer median MIS')
plt.show()
'''

########################## ALL LAYERS ##########################
def get_mis_vals(layer):
    if layer == "mixed3a":
        original_layer_mis_median = 0.834
        epoch = 7
    elif layer == "mixed3b":
        original_layer_mis_median = 0.862
        epoch = 6
    elif layer == "mixed4a":
        original_layer_mis_median = 0.888
        epoch = 6
    elif layer == "mixed4b":
        original_layer_mis_median = 0.922
        epoch = 6
    elif layer == "mixed4c":
        original_layer_mis_median = 0.944
        epoch = 5
    elif layer == "mixed4d":
        original_layer_mis_median = 0.953
        epoch = 7
    elif layer == "mixed4e":
        original_layer_mis_median = 0.954
        epoch = 9
    elif layer == "mixed5a":
        original_layer_mis_median = 0.950
        epoch = 5
    elif layer == "mixed5b":
        original_layer_mis_median = 0.899
        epoch = 12

    if layer == "mixed3a":
        dead_neurons_steps = 626
        exp_fac = 8
        lam = 5.0
    else:
        dead_neurons_steps = 625
        exp_fac = 4
        lam = 0.1
    #'''

    sae_eval_results = rf"C:\Users\Jasper\Downloads\Master thesis\code\evaluation_results\inceptionv1\imagenet\{layer}_inceptionv1_1_0.001_512_sgd_sae_mlp_{dead_neurons_steps}_sae_eval_results.csv"
    sae_eval_results = pd.read_csv(sae_eval_results)

    #'''
    # only keep those rows where mis != 0 (missing datapoint due to run not terminating or taking too long)
    sae_eval_results = sae_eval_results[sae_eval_results['median_mis'] != 0]
    # sort by epochs
    sae_eval_results = sae_eval_results.sort_values(by='epochs')
    # get the mis of the sae at the specified checkpoint epoch
    mis_epoch = sae_eval_results[sae_eval_results['epochs'] == epoch]['median_mis']

    return sae_eval_results, original_layer_mis_median, mis_epoch

# plot mis of one layer over epochs
'''
layer = "mixed4d"
sae_eval_results, original_layer_mis_median, _ = get_mis_vals(layer)
plt.plot(sae_eval_results['epochs'], sae_eval_results['median_mis'])
plt.xlabel('Epoch')
plt.ylabel('Median MIS')
#plt.ylim(0.82, 0.88)
#plt.xlim(0,10)
plt.title(f'Median MIS of SAE on {layer} with exp fac {exp_fac} and lambda {lam}')
plt.axhline(y=original_layer_mis_median, color='b', linestyle='--', label='Original layer median MIS')
plt.show()
'''

# plot mis of all layers at checkpoint epochs
layers = ["mixed3a", "mixed3b", "mixed4a", "mixed4b", "mixed4c", "mixed4d", "mixed4e", "mixed5a", "mixed5b"]
original_layer_mis_medians = []
sae_layer_mis_medians = []
for layer in layers:
    _, original_layer_mis_median, mis_epoch = get_mis_vals(layer)
    original_layer_mis_medians.append(original_layer_mis_median)
    sae_layer_mis_medians.append(mis_epoch.values[0])
# change the x axis tick labels: "mixed3a" -> "3a"
layers = [layer.replace("mixed", "") for layer in layers]
plt.plot(layers, original_layer_mis_medians, label='Model')
plt.plot(layers, sae_layer_mis_medians, label='SAE')
fontsize = 15
plt.xticks(fontsize=fontsize-2)
plt.yticks(fontsize=fontsize-2)
plt.xlabel('Layer', fontsize=fontsize, labelpad=10)
plt.ylabel('Median MIS', fontsize=fontsize, labelpad=10)
plt.legend(fontsize=fontsize-2)
plt.tight_layout()
plt.savefig(r"C:\Users\Jasper\Downloads\Master thesis\code\evaluation_results\inceptionv1\imagenet\mis_over_layers.png", dpi=300)
#plt.show()