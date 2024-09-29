import os
import matplotlib.pyplot as plt
import numpy as np

from evaluate_model import ModelEvaluator
from utils import load_data_aux

directory_path = r'C:\Users\Jasper\Downloads\Master thesis\Code'
model_name = 'custom_mlp_1'
dataset_name = 'cifar_10'
weights_folder_path = os.path.join(directory_path, 'model_weights', model_name, dataset_name)

model_evaluator = ModelEvaluator('single_model', 
                                 model_name, 
                                 dataset_name, 
                                 weights_folder_path=weights_folder_path)

_, valloader, img_size, class_names = load_data_aux(dataset_name, data_dir=None, layer_name=None)

input_images, target_ids, output = model_evaluator.get_model_output('first_batch', weights_folder_path)
scores, predicted_classes = model_evaluator.get_classification(output)

number_of_images = 10 # show only the first n images, 
# for showing all images in the batch use len(predicted_classes)
fig, axes = plt.subplots(1, number_of_images, figsize=(15, 3))

for i in range(number_of_images):
    # Unnormalize the image
    img = input_images[i] / 2 + 0.5
    npimg = img.numpy()

    # Display the image with true label, predicted label, and score
    axes[i].imshow(np.transpose(npimg, (1, 2, 0)))
    title = f'True: {class_names[target_ids[i]]}\nPredicted: {predicted_classes[i]} ({scores[i].item():.1f}%)'
    # alternatively: scores.detach().numpy()[i]
    axes[i].set_title(title, fontsize=8)
    axes[i].axis('off')
    
plt.subplots_adjust(wspace=0.5)  # Adjust space between images
plt.show()