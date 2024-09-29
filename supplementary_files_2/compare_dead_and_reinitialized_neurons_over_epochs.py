dead_neurons_eval_epoch_8_txt = r"C:\Users\Jasper\Downloads\Master thesis\code\evaluation_results\inceptionv1\imagenet\indices_of_dead_neurons\mixed3a_inceptionv1_1_0.001_512_sgd_sae_mlp_0.001_256_constrained_adam_8_5.0_195_epoch_8_epoch_batch_idx_5002.txt"
dead_neurons_eval_epoch_7_txt = r"C:\Users\Jasper\Downloads\Master thesis\code\evaluation_results\inceptionv1\imagenet\indices_of_dead_neurons\mixed3a_inceptionv1_1_0.001_512_sgd_sae_mlp_0.001_256_constrained_adam_8_5.0_195_epoch_7_epoch_batch_idx_5002.txt"
dead_neurons_eval_epoch_6_txt = r"C:\Users\Jasper\Downloads\Master thesis\code\evaluation_results\inceptionv1\imagenet\indices_of_dead_neurons\mixed3a_inceptionv1_1_0.001_512_sgd_sae_mlp_0.001_256_constrained_adam_8_5.0_195_epoch_6_epoch_batch_idx_5002.txt"

re_initialized_neurons_train_epoch_8_txt = r"C:\Users\Jasper\Downloads\Master thesis\code\evaluation_results\inceptionv1\imagenet\indices_of_re_initialized_neurons\mixed3a_inceptionv1_1_0.001_512_sgd_sae_mlp_0.001_256_constrained_adam_8_5.0_195_epoch_8_train_batch_idx_40015_epoch_batch_idx_5001.txt"

with open(dead_neurons_eval_epoch_8_txt) as f:
    dead_neurons_eval_epoch_8 = f.readlines()
    dead_neurons_eval_epoch_8 = [int(line.strip()) for line in dead_neurons_eval_epoch_8]

with open(dead_neurons_eval_epoch_7_txt) as f:
    dead_neurons_eval_epoch_7 = f.readlines()
    dead_neurons_eval_epoch_7 = [int(line.strip()) for line in dead_neurons_eval_epoch_7]

with open(dead_neurons_eval_epoch_6_txt) as f:
    dead_neurons_eval_epoch_6 = f.readlines()
    dead_neurons_eval_epoch_6 = [int(line.strip()) for line in dead_neurons_eval_epoch_6]

with open(re_initialized_neurons_train_epoch_8_txt) as f:
    re_initialized_neurons_train_epoch_8 = f.readlines()
    re_initialized_neurons_train_epoch_8 = [int(line.strip()) for line in re_initialized_neurons_train_epoch_8]

print("Number of dead neurons in eval epoch 6: ", len(dead_neurons_eval_epoch_6))
print("Number of dead neurons in eval epoch 7: ", len(dead_neurons_eval_epoch_7))
print("Number of dead neurons in eval epoch 8: ", len(dead_neurons_eval_epoch_8))
print("-------------------------------------")
print("Number of re-initialized neurons in train epoch 8: ", len(re_initialized_neurons_train_epoch_8))
print("Number of same dead neurons in eval epoch 7 and train epoch 8: ", len(set(dead_neurons_eval_epoch_7).intersection(re_initialized_neurons_train_epoch_8)))
print("-------------------------------------")
print("Number of same dead neurons in eval epochs 6 and 7: ", len(set(dead_neurons_eval_epoch_6).intersection(dead_neurons_eval_epoch_7)))
print("Number of same dead neurons in eval epochs 7 and 8: ", len(set(dead_neurons_eval_epoch_7).intersection(dead_neurons_eval_epoch_8)))
print("-------------------------------------")