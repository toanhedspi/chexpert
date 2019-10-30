from __future__ import print_function, division
import os
import torch
import pandas as pd # For csv file reading
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision import transforms, utils

class TrainingDatasetClass(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None, gray_scale = False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.training_datas = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.gray_scale = gray_scale

    def __len__(self):
        return len(self.training_datas)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.training_datas.iloc[idx, 0])

        # Split info                        
        image = io.imread(img_name)
        sex = self.training_datas.iloc[idx, 1]
        age = self.training_datas.iloc[idx, 2]
        anatomy = self.training_datas.iloc[idx, 3]
        view = self.training_datas.iloc[idx, 4]
        diagnoses = self.training_datas.iloc[idx, 5:]

        diagnoses = np.array([diagnoses])
        diagnoses = diagnoses.astype('float').reshape(-1, 14)
        sample = {'image': image, 'sex': sex, age: 'age', anatomy: 'anatomy', view: 'view', 'diagnoses': diagnoses}

        if self.transform:
            sample = self.transform(sample)

        return sample

