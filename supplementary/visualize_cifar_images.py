import matplotlib.pyplot as plt
import numpy as np

''' 
# This can be done as an alternative to installing subdirectories as packages
# through the setup.py and the install.sh files
import sys
parent_dir = 'C:\\Users\\Jasper\\Downloads\\Master thesis\\Code'
sys.path.append(parent_dir)  # Add the parent directory to the Python path
'''

from utils import load_data_aux

def show_images(images, labels, classes):
    batch_size = len(images)
    fig, axes = plt.subplots(1, batch_size, figsize=(15, 3))
    
    for i in range(batch_size):
        # Unnormalize the image
        img = images[i] / 2 + 0.5
        npimg = img.numpy()

        # Display the image with smaller labels
        axes[i].imshow(np.transpose(npimg, (1, 2, 0)))
        axes[i].set_title(classes[labels[i]], fontsize=8)
        axes[i].axis('off')

    plt.subplots_adjust(wspace=0.5)  # Adjust space between images
    plt.show()

# Load CIFAR-10 dataset
dataset_name = 'cifar_10'
trainloader, _, _, classes = load_data_aux(dataset_name, data_dir=None, layer_name=None)

# Get a batch of random training images
dataiter = iter(trainloader)
images, labels = next(dataiter)

# Show the images with smaller labels and more space
show_images(images, labels, classes)