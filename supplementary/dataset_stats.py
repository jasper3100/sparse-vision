import torch
'''
def print_dataset_stats(train_dataset):     
    imgs = [item[0] for item in train_dataset] # item[0] and item[1] are image and its label
    imgs = torch.stack(imgs, dim=0).numpy()

    # calculate mean over each channel (r,g,b)
    mean_r = imgs[:,0,:,:].mean()
    mean_g = imgs[:,1,:,:].mean()
    mean_b = imgs[:,2,:,:].mean()
    print(mean_r,mean_g,mean_b)

    # calculate std over each channel (r,g,b)
    std_r = imgs[:,0,:,:].std()
    std_g = imgs[:,1,:,:].std()
    std_b = imgs[:,2,:,:].std()
    print(std_r,std_g,std_b)

    min_r, min_g, min_b = imgs[:, 0, :, :].min(), imgs[:, 1, :, :].min(), imgs[:, 2, :, :].min()
    max_r, max_g, max_b = imgs[:, 0, :, :].max(), imgs[:, 1, :, :].max(), imgs[:, 2, :, :].max()

    print("Min pixel values (r, g, b):", min_r, min_g, min_b)
    print("Max pixel values (r, g, b):", max_r, max_g, max_b)
'''
def print_dataset_stats(train_dataset):
    # Get the number of channels from the first image in the dataset
    num_channels = train_dataset[0][0].shape[0]

    imgs = [item[0] for item in train_dataset]  # item[0] and item[1] are image and its label
    imgs = torch.stack(imgs, dim=0).numpy()

    # Calculate mean over each channel
    means = [imgs[:, i, :, :].mean() for i in range(num_channels)]
    print("Mean pixel values for each channel:", means)

    # Calculate std over each channel
    stds = [imgs[:, i, :, :].std() for i in range(num_channels)]
    print("Std deviation of pixel values for each channel:", stds)

    # Calculate min and max over each channel
    mins = [imgs[:, i, :, :].min() for i in range(num_channels)]
    maxs = [imgs[:, i, :, :].max() for i in range(num_channels)]

    print("Min pixel values for each channel:", mins)
    print("Max pixel values for each channel:", maxs)
