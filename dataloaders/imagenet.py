import os
import torch
from torchvision import datasets, transforms

# DOES THIS EVEN WORK???

class ImageNetDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        # Read the list of image files for the specified split
        split_file = os.path.join(root_dir, 'ImageSets', 'CLS-LOC', f'{split}.txt')
        with open(split_file, 'r') as f:
            self.image_files = f.read().strip().split('\n')

        # Set the path to the images
        self.image_paths = [os.path.join(root_dir, 'Data', 'CLS-LOC', split, f'{img}.JPEG') for img in self.image_files]

        # Map class names to numerical labels
        self.class_to_idx = {folder: idx for idx, folder in enumerate(sorted(os.listdir(os.path.join(root_dir, 'Data', 'CLS-LOC', split))))}

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = datasets.folder.default_loader(img_path)
        if self.transform:
            image = self.transform(image)
        label = self.class_to_idx[img_path.split('/')[-2]]
        return image, label