train_batch_idx = 0
dead_neurons_steps = 9912

# print all multiples of 9912 up to 100000
print([i for i in range(0, 100000, dead_neurons_steps)])

while train_batch_idx < 100000:
    train_batch_idx += 1
    if (train_batch_idx - 1) % dead_neurons_steps == 0 and ((train_batch_idx - 1) // dead_neurons_steps) % 2 == 0 and (train_batch_idx - 1) != 0:
        print("Re-initialize", train_batch_idx)

    elif (train_batch_idx == dead_neurons_steps) or (train_batch_idx > dead_neurons_steps and train_batch_idx % dead_neurons_steps == 0 and (train_batch_idx // dead_neurons_steps) % 2 == 1):
        print("Wait", train_batch_idx)
