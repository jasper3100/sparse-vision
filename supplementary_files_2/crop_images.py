from PIL import Image
# open image in paint.net to find out the pixels of the box to crop

# to cut plot from W&B generated images
generic_wandb_crop_box = (46, 175, 1694, 1264) # (left, upper, right, lower)

# REC LOSS VS SPARSITY PLOTS
layer = "mixed3a"
version = ["v1", "v2"]#, "v3"]
on = False # turn this computation on or off

if on:
    for v in version:
        file_path = fr"C:\Users\Jasper\Downloads\Master thesis\code\evaluation_results\inceptionv1\imagenet\{layer}_inceptionv1_1_0.001_512_sgd_sae_mlp_300_sae_eval_results_plot_nrmse_loss_many_epochs_{v}.png"
        image = Image.open(file_path)

        crop_box = (243, 243, 3794, 2079) # (left, upper, right, lower)

        # Crop the image
        cropped_image = image.crop(crop_box)

        # save the image
        cropped_image.save(file_path.replace(".png", "_cropped.png"))

# EVAL L1 LOSS SMALL-SCALE SETTING 
layers = ["Mixed3a", "Mixed3b", "Mixed4a", "Mixed4d"]
metric = "L1 loss grouped by exp fac" #"Density" # "L1 loss"
on = False

if on:
    for layer in layers:
        file_path = fr"C:\Users\Jasper\Downloads\GoogLeNet ImageNet Small-scale setting Eval {metric} {layer}.png"
        image = Image.open(file_path)

        # cutting the legend
        #crop_box = (1716, 175, 1961, 1264)
        cropped_image = image.crop(generic_wandb_crop_box)
        cropped_image.save(file_path.replace(".png", "_cropped.png"))

# RE-INIT SCENARIOS
version = ["a", "b", "c"]
on = False # turn this computation on or off

if on:
    for v in version:
        file_path = fr"C:\Users\Jasper\Downloads\{v}.png"
        image = Image.open(file_path)

        crop_box = (43, 156, 1716, 1286) # (left, upper, right, lower)

        # Crop the image
        cropped_image = image.crop(crop_box)

        # save the image
        cropped_image.save(file_path.replace(".png", "_cropped.png"))


# CHANNEL ACTIVATION FREQUENCY HISTOGRAMS
epochs = [0, 10, 20, 30, 40, 50]
on = False # turn this computation on or off

if on:
    for epoch in epochs:
        file_path = fr"C:\Users\Jasper\Downloads\mixed3a_inceptionv1_1_0.001_512_sgd_gated_sae_0.001_256_constrained_adam_4_5.0_300_channel_frequency_histograms_epoch_{epoch}.png"
        image = Image.open(file_path)

        crop_box = (35,53,579,452)

        # Crop the image
        cropped_image = image.crop(crop_box)

        # save the image
        cropped_image.save(file_path.replace(".png", "_cropped.png"))


# INCREASING LAMBDA GRADUALLY
names = ["Train rec. loss", "Eval Density", "Train L1 Loss", "Train perc of dead neurons"]
on = True # turn this computation on or off

if on:
    for name in names:
        file_path = fr"C:\Users\Jasper\Downloads\GoogLeNet ImageNet Full-scale setting {name} Mixed3a increasing lambda gradually.png"
        image = Image.open(file_path)

        cropped_image = image.crop(generic_wandb_crop_box)

        # save the image
        cropped_image.save(file_path.replace(".png", "_cropped.png"))
