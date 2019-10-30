from __future__ import print_function, division
from TrainingDatasetClass import TrainingDatasetClass
import os
import torch
from torchvision import transforms, datasets
import pandas as pd # For csv file reading
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# class ToTensor(object):
#     """Convert ndarrays in sample to Tensors."""

#     def __call__(self, sample):
#         image, diagnoses = sample['image'], sample['diagnoses']

#         # swap color axis because
#         # numpy image: H x W x C
#         # torch image: C X H X W

#         return {'image': torch.from_numpy(np.array([image])),
#                 'diagnoses': torch.from_numpy(diagnoses)}

# Read file data
training_datas = TrainingDatasetClass(csv_file='CheXpert-v1.0-small/train.csv',
                                root_dir='')

testing_datas = TrainingDatasetClass(csv_file='CheXpert-v1.0-small/valid.csv',
                                root_dir='')

# SAMPLE IS HERE
sample = training_datas[0]

def show_image(image):
    """Show image with landmarks"""
    plt.imshow(image,cmap='gray')

    plt.pause(5)  # pause a bit so that plots are updated
    plt.close()

# Convert to tensor
composed = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5],[0.5])])

tramsformed_sample = composed(sample['image'])

# Initiate trainloader, testloader
train_loader = torch.utils.data.DataLoader(training_datas, batch_size=16,
                                        shuffle=True, num_workers=2)

test_loader = torch.utils.data.DataLoader(testing_datas, batch_size=16,
                                        shuffle=False, num_workers=2)

# Constant for classes
classes = ('No Finding', 'Enlarged Cardiomediastinum', 'Lung Opacity', 'Lung Lesion', 'Edema',
        'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices')

