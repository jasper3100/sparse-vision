import imageio
import numpy as np
import os
import urllib.request
import zipfile

from collections import defaultdict
from torch.utils.data import Dataset

from tqdm.autonotebook import tqdm

# FROM HERE, with minor modifications: https://gist.github.com/z-a-f/b862013c0dc2b540cf96a123a6766e54#file-tin-py-L143-L201

dir_structure_help = r"""
TinyImageNetPath
├── test
│   └── images
│       ├── test_0.JPEG
│       ├── t...
│       └── ...
├── train
│   ├── n01443537
│   │   ├── images
│   │   │   ├── n01443537_0.JPEG
│   │   │   ├── n...
│   │   │   └── ...
│   │   └── n01443537_boxes.txt
│   ├── n01629819
│   │   ├── images
│   │   │   ├── n01629819_0.JPEG
│   │   │   ├── n...
│   │   │   └── ...
│   │   └── n01629819_boxes.txt
│   ├── n...
│   │   ├── images
│   │   │   ├── ...
│   │   │   └── ...
├── val
│   ├── images
│   │   ├── val_0.JPEG
│   │   ├── v...
│   │   └── ...
│   └── val_annotations.txt
├── wnids.txt
└── words.txt
"""

def download_and_unzip(URL, root_dir):
    # Check if the root directory exists, and create it if not
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    # Define the file name based on the URL
    file_name = os.path.join(root_dir, URL.split('/')[-1])

    # Download the file if it doesn't exist
    if not os.path.exists(file_name):
        print(f"Downloading {URL}...")
        urllib.request.urlretrieve(URL, file_name)

    # Unzip the downloaded file
    with zipfile.ZipFile(file_name, 'r') as zip_ref:
        print(f"Extracting files to {root_dir}...")
        zip_ref.extractall(root_dir)

    print("Download and extraction complete.")

# Example usage:
# download_and_unzip('http://cs231n.stanford.edu/tiny-imagenet-200.zip', 'datasets/tiny_imagenet_2')


def _add_channels(img, total_channels=3):
  '''
  Some images are greyscale (no channel dimension, since there is only one channel)
  Some images are RGB (there is a channel dimension, since there are 3 channels)
  '''
  while len(img.shape) < 3:  # third axis is the channels
    img = np.expand_dims(img, axis=-1)
  while(img.shape[-1]) < 3:
    img = np.concatenate([img, img[:, :, -1:]], axis=-1)
  return img

"""Creates a paths datastructure for the tiny imagenet.

Args:
  root_dir: Where the data is located
  download: Download if the data is not there

Members:
  label_id:
  ids:
  nit_to_words:
  data_dict:

"""
class TinyImageNetPaths:
  def __init__(self, root_dir, download=False):
    if download:
      download_and_unzip('http://cs231n.stanford.edu/tiny-imagenet-200.zip',
                         root_dir)
    # Create the root directory if it doesn't exist
    if not os.path.exists(root_dir):
      os.makedirs(root_dir)

    train_path = os.path.join(root_dir, 'train')
    val_path = os.path.join(root_dir, 'val')
    test_path = os.path.join(root_dir, 'test')

    wnids_path = os.path.join(root_dir, 'wnids.txt')
    words_path = os.path.join(root_dir, 'words.txt')

    self._make_paths(train_path, val_path, test_path,
                     wnids_path, words_path)

  def _make_paths(self, train_path, val_path, test_path,
                  wnids_path, words_path):
    self.ids = []
    with open(wnids_path, 'r') as idf:
      for nid in idf:
        nid = nid.strip()
        self.ids.append(nid)
    self.nid_to_words = defaultdict(list)
    with open(words_path, 'r') as wf:
      for line in wf:
        nid, labels = line.split('\t')
        labels = list(map(lambda x: x.strip(), labels.split(',')))
        self.nid_to_words[nid].extend(labels)

    self.paths = {
      'train': [],  # [img_path, id, nid, box]
      'val': [],  # [img_path, id, nid, box]
      'test': []  # img_path
    }

    # Get the test paths
    self.paths['test'] = list(map(lambda x: os.path.join(test_path, x),
                                      os.listdir(test_path)))
    # Get the validation paths and labels
    with open(os.path.join(val_path, 'val_annotations.txt')) as valf:
      for line in valf:
        fname, nid, x0, y0, x1, y1 = line.split()
        fname = os.path.join(val_path, 'images', fname)
        bbox = int(x0), int(y0), int(x1), int(y1)
        label_id = self.ids.index(nid)
        self.paths['val'].append((fname, label_id, nid, bbox))

    # Get the training paths
    train_nids = os.listdir(train_path)
    for nid in train_nids:
      anno_path = os.path.join(train_path, nid, nid+'_boxes.txt')
      imgs_path = os.path.join(train_path, nid, 'images')
      label_id = self.ids.index(nid)
      with open(anno_path, 'r') as annof:
        for line in annof:
          fname, x0, y0, x1, y1 = line.split()
          fname = os.path.join(imgs_path, fname)
          bbox = int(x0), int(y0), int(x1), int(y1)
          self.paths['train'].append((fname, label_id, nid, bbox))

"""Datastructure for the tiny image dataset.

Args:
  root_dir: Root directory for the data
  mode: One of "train", "test", or "val"
  preload: Preload into memory
  load_transform: Transformation to use at the preload time
  transform: Transformation to use at the retrieval time
  download: Download the dataset

Members:
  tinp: Instance of the TinyImageNetPaths
  img_data: Image data
  label_data: Label data
"""
class TinyImageNetDataset(Dataset):
  def __init__(self, root_dir, mode='train', preload=True, load_transform=None,
               transform=None, download=False, max_samples=None):
    tinp = TinyImageNetPaths(root_dir, download)
    self.mode = mode
    self.label_idx = 1  # from [image, id, nid, box]
    self.preload = preload
    self.transform = transform
    self.transform_results = dict()

    # I ADJUSTED THE BELOW LINE
    self.IMAGE_SHAPE = (3,64,64) #(64, 64, 3)

    self.img_data = []
    self.label_data = []

    self.max_samples = max_samples
    self.samples = tinp.paths[mode]
    self.samples_num = len(self.samples)

    if self.max_samples is not None:
      self.samples_num = min(self.max_samples, self.samples_num)
      self.samples = np.random.permutation(self.samples)[:self.samples_num]

    if self.preload:
      load_desc = "Preloading {} data...".format(mode)
      self.img_data = np.zeros((self.samples_num,) + self.IMAGE_SHAPE,
                               dtype=np.float32)
      self.label_data = np.zeros((self.samples_num,), dtype=np.int)
      for idx in tqdm(range(self.samples_num), desc=load_desc):
        s = self.samples[idx]
        img = imageio.v3.imread(s[0])
        img = _add_channels(img)
        # I ADDED THE BELOW LINE TO GET THE SHAPE (channel, width, height) instead of (width, height, channel)
        img = np.transpose(img, (2, 0, 1)) 
        self.img_data[idx] = img
        if mode != 'test':
          self.label_data[idx] = s[self.label_idx]

      if load_transform:
        for lt in load_transform:
          result = lt(self.img_data, self.label_data)
          self.img_data, self.label_data = result[:2]
          if len(result) > 2:
            self.transform_results.update(result[2])

  def __len__(self):
    return self.samples_num

  def __getitem__(self, idx):
    if self.preload:
      img = self.img_data[idx]
      lbl = None if self.mode == 'test' else self.label_data[idx]
    else:
      s = self.samples[idx]
      img = imageio.v3.imread(s[0])
      img = _add_channels(img) 
      # I ADDED THE BELOW LINE TO GET THE SHAPE (channel, width, height) instead of (width, height, channel)
      img = np.transpose(img, (2, 0, 1))
      lbl = None if self.mode == 'test' else s[self.label_idx]
    # I ADDED THE BELOW LINE
    # convert data type from uint8 to float32 for the model to work
    img = np.float32(img)
    # Alternatively when iterating through the batches of the dataloader, one can do:
    # img = img.to(torch.float32)
    sample = {'image': img, 'label': lbl}

    if self.transform:
      sample = self.transform(sample)
    return sample

# The below code was moved to other scripts, but is just provided for reference here
# and to test the data loading in one script if necessary

'''
root_dir = 'datasets/tiny-imagenet-200'
train_dataset = TinyImageNetDataset(root_dir, mode='train', preload=False, download=False) # Download=True only if dataset not yet downloaded

from torch.utils.data import DataLoader
from model import model

batch_size = 32
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

for batch in train_dataloader:
    inputs, labels = batch['image'], batch['label']
    
    outputs = model(inputs)

    print(outputs.shape)
    print(labels.shape)
    #print(labels)
    break
'''

'''
for epoch in range(num_epochs):
    for batch in dataloader:
        inputs, labels = batch['image'], batch['label']

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
'''