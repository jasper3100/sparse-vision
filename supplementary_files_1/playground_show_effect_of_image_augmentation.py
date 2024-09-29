import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os

directory_path = 'C:\\Users\\Jasper\\Downloads\\Master thesis\\Code'

batch_size = 64
train_shuffle = True
eval_shuffle = False
drop_last = True

root_dir='datasets/cifar-10'
# if root_dir does not exist, download the dataset
root_dir=os.path.join(directory_path, root_dir)
# The below works on the cluster
#root_dir='/lustre/home/jtoussaint/master_thesis/datasets/cifar-10'
download = not os.path.exists(root_dir)
if not os.path.exists(root_dir):
    os.makedirs(root_dir, exist_ok=True)

train_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
])
#test_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
# no normalization during testing???

# dataloader argument num_workers = 2 ? 
# separate between train and test transforms here...
train_dataset = torchvision.datasets.CIFAR10(root_dir, train=True, download=download, transform=train_transform)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=train_shuffle, drop_last=drop_last)
#val_dataset = torchvision.datasets.CIFAR10(root_dir, train=False, download=download, transform=test_transform)
#val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=eval_shuffle, drop_last=drop_last)

# Get a batch of training data
data_iter = iter(train_dataloader)
images, labels = data_iter.next()

augmentations_list = [ #torchvision.transforms.Compose([
                #torchvision.transforms.ToPILImage(),
                torchvision.transforms.RandomCrop(size=(32,32)),#(data_resolution, padding=int(data_resolution * 0.125), fill=data_mean_int),
                torchvision.transforms.RandomHorizontalFlip(),
                #common.autoaugment.CIFAR10Policy(fillcolor=data_mean_int),
                #torchvision.transforms.AutoAugment.CIFAR10Policy(fillcolor=data_mean_int),
                torchvision.transforms.AutoAugment(policy=torchvision.transforms.autoaugment.AutoAugmentPolicy.CIFAR10, fill=int(125)), # fill needs to be an integer??? or tuple of integers???
                #torchvision.transforms.ToTensor(),
                #transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
                #common.autoaugment.CutoutAfterToTensor(n_holes=1, length=cutout, fill_color=data_mean),
                torchvision.transforms.RandomErasing()#(value=data_mean_int),
            ]#)

#inputs = images.type(torch.uint8)
inputs=images
# Randomly choose one augmentation from the list for each image in the batch
random_augmentation = transforms.RandomChoice(augmentations_list)
# Apply the randomly chosen augmentation to each image in the batch
inputs = torch.stack([random_augmentation(img) for img in inputs])
# apply toTensor to each img
#inputs = torch.stack([transforms.ToTensor()(img) for img in inputs])
# apply normalization to each img
#inputs = inputs.type(torch.float32)
#inputs = torch.stack([transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))(img) for img in inputs])
#'''

# Unnormalize the images (if normalization was applied)
#unnormalize = transforms.Compose([
#    transforms.Normalize((-1, -1, -1), (2, 2, 2))  # Inverse of the normalization applied
#])

# Function to display images
def imshow(img):
    #img = unnormalize(img)  # Unnormalize the image
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')

# Plot images
num_images_to_show = 4
plt.figure(figsize=(10, 4))
for i in range(num_images_to_show):
    plt.subplot(1, num_images_to_show, i + 1)
    imshow(inputs[i])
    plt.title(f'Class: {labels[i].item()}')

plt.show()
 