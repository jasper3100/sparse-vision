import os
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import torch

# CREATING AN AUGMENTED DATASET --> BUT THIS IS NOT HOW DATA AUGMENTATION WORKS. IT IS APPLIED
# BATCHWISE RANDOMLY. WE DONT CREATE A LARGER DATASET OF FIXED AUGMENTED IMAGES!!!
# FURTHER DOWN BELOW IS AN EXAMPLE OF HOW TO APPLY DATA AUGMENTATION TO A BATCH OF IMAGES

class AugmentedCIFAR10(Dataset):
    def __init__(self, root_dir, train=True, download=True, transform=None):
        self.original_dataset = torchvision.datasets.CIFAR10(root_dir, train=train, download=download, transform=transform)
        self.augmentations = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.RandomCrop(28, padding=4)] # padding is applied to the image before the crop
        # the original image size is 32x32, so we crop it to 28x28, could try 24x24 etc. but the objects are usually quite large
        # hence 24x24 might cut too much of the image?

    def __len__(self):
        return len(self.original_dataset) * (len(self.augmentations) + 1)

    def __getitem__(self, idx):
        original_idx = idx % len(self.original_dataset)
        # module returns the remainder of the division of idx by len(self.original_dataset)
        # For example, if length of the original dataset is 50,000, and the index is 65,000, then the original_idx is 15,000
        is_original = idx < len(self.original_dataset)
        # the last image of the original dataset (of length 50,000) is at index 49,999, hence an index larger than that is 
        # not part of the original dataset
        if is_original:
            return self.original_dataset[idx]
        else:
            augmentation_index = idx // len(self.original_dataset) - 1
            # for example, an image with index 75,000 (where length of orig. dataset is 50,000) -> augmentation_index = 1 - 1 = 0
            # an image with index 125,000 -> augmentation_index = 2 - 1 = 1
            original_image, label = self.original_dataset[original_idx]
            augmented_image = self.augmentations[augmentation_index](self.original_dataset[original_idx][0])
            return augmented_image, label


directory_path = 'C:\\Users\\Jasper\\Downloads\\Master thesis\\Code'
batch_size = 64
root_dir = 'datasets/cifar-10'
root_dir = os.path.join(directory_path, root_dir)

download = not os.path.exists(root_dir)
if not os.path.exists(root_dir):
    os.makedirs(root_dir, exist_ok=True)

# Define data transformations for the original dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load the augmented dataset
augmented_dataset = AugmentedCIFAR10(root_dir, train=True, download=download, transform=transform)
augmented_dataloader = DataLoader(augmented_dataset, batch_size=batch_size, shuffle=False)
category_names = augmented_dataset.original_dataset.classes
num_samples = len(augmented_dataloader.dataset)

# Show an image
'''
image, label = augmented_dataset[0] # image at index 0
# unnormalize the image
image = image / 2 + 0.5
plt.imshow(image.permute(1, 2, 0))
plt.title(category_names[label])
plt.show()
'''

# load one batch of images
images, labels = next(iter(augmented_dataloader))

# print the shape of images
print(images.shape)

'''
# show the first 5 images
plt.figure(figsize=(10, 5))
for i, image in enumerate(images[:5]):
    plt.subplot(1, 5, i + 1)
    # unnormalize the image
    image = image / 2 + 0.5
    plt.imshow(image.permute(1, 2, 0))
    plt.axis('off')
    plt.title(category_names[labels[i]])
plt.show()
'''

# We define a list of data augmentations
augmentations_list = [
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomResizedCrop((32, 32), scale=(0.8, 1.0), ratio=(0.8, 1.2))]
# Randomly choose one augmentation from the list for each image in the batch
random_augmentation = transforms.RandomChoice(augmentations_list)
# Apply the randomly chosen augmentation to each image in the batch
images = torch.stack([random_augmentation(img) for img in images])

# Show the first 5 images after applying the random augmentation
plt.figure(figsize=(10, 5))
for i, image in enumerate(images[:5]):
    plt.subplot(1, 5, i + 1)
    # unnormalize the image
    image = image / 2 + 0.5
    plt.imshow(image.permute(1, 2, 0))
    plt.axis('off')
    plt.title(category_names[labels[i]])
plt.show()