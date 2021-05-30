"""
epoch = 1 forward and backward pass of ALL training samples
batch_size = number of training samples in one forward & backward pass
number of iterations = number of passes, each pass using [batch_size] number of samples
e.g. 100 samples, batch_size = 20 --> 100/20 = 5 iterations for 1 epoch
"""

'''
Transforms can be applied to PIL images, tensors, ndarrays, or custom data
during creation of the DataSet
complete list of built-in transforms: 
https://pytorch.org/docs/stable/torchvision/transforms.html
On Images
---------
CenterCrop, Grayscale, Pad, RandomAffine
RandomCrop, RandomHorizontalFlip, RandomRotation
Resize, Scale
On Tensors
----------
LinearTransformation, Normalize, RandomErasing
Conversion
----------
ToPILImage: from tensor or ndrarray
ToTensor : from numpy.ndarray or PILImage
Generic
-------
Use Lambda 
Custom
------
Write own class
Compose multiple Transforms
---------------------------
composed = transforms.Compose([Rescale(256),
                               RandomCrop(224)])
'''

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math

class WineDataset(Dataset):
    def __init__(self, transform=None):
        xy = np.loadtxt('wine.csv', delimiter=',', dtype=np.float32, skiprows=1)
        self.n_samples = xy.shape[0]

        self.x = xy[:,1:]
        self.y = xy[:,[0]]

        self.transform = transform

    def __getitem__(self, index):
        sample = self.x[index], self.y[index]

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return self.n_samples

class ToTensor():
    def __call__(self, sample):
        inputs, targets = sample
        return torch.from_numpy(inputs), torch.from_numpy(targets)

class MulTransform:
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, sample):
        inputs, target = sample
        inputs *= self.factor
        return inputs, target

dataset = WineDataset(ToTensor())
dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True, num_workers=2)

#Transformed data
dataset_t = WineDataset()
first_sample = dataset_t[0]
features, labels = first_sample
print(features)
print(type(features), type(labels))

composed = torchvision.transforms.Compose([ToTensor(), MulTransform(4)])
dataset_m = WineDataset(transform=composed)
first_sample = dataset_m[0]
features, labels = first_sample
print(features)
print(type(features), type(labels))

#training loop
num_epoch = 2
total_samples = len(dataset)
num_iterations = math.ceil(total_samples/4)
print(total_samples, num_iterations)

if __name__ == '__main__':
    for epoch in range(num_epoch):
        for i, (inputs, labels) in enumerate(dataloader):
            #forwarw and barkward, update
            if (i+1) %5 ==0:
                print(f'epoch {epoch+1}/{num_epoch}, step {i+1}/{num_iterations}, inputs {inputs.shape}')

