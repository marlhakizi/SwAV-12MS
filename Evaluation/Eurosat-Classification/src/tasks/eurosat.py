from PIL import Image
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


import torch
from torch.utils.data import Dataset,DataLoader,random_split
import torchvision.transforms as transforms
import albumentations as A

from torch import nn
from torch.nn import functional as F


class Eurosat(Dataset):
    """
    This class takes in the folder with RGB Eurosat data and the file names after train/test split 
    Return the image tensor and the associated class of that image
    """
    def __init__(self, path, files, transform= None, augmentations=None):
        self.img_path = path   # Path to images
        self.files = files  # list of filenames in the file path
        self.transform = transforms.Compose([transform, 
        # this transformation here is to make sure we have 
        transforms.CenterCrop(size = 64),
        transforms.Resize(size = 224)])
        self.aug = augmentations

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        # Load the image 
        img_name = self.img_path + str(self.files[idx]) # this will be something like '/scratch/yc506/eurosat_rgb/Residential_2054.tif'
        # extract the image class with the name  
        name_class = str(self.files[idx]).lower().split('_')[0] # this will give us something like 'forest'
        class_map = {'annualcrop': 0 , 'forest':1, 'herbaceousvegetation': 2, 'highway':3, 'industrial':4, 'pasture': 5, 'permanentcrop':6,
                     'residential': 7, 'river': 8, 'sealake': 9
                      }
        img_class = class_map[name_class] # assign numerical class to each image

        # Apply albumentations augmentations
        # before applying the standard transform
        if self.aug:
            # Must convert to numpy array for
            # this to work nicely with albumentations
            # cv2 is the recommended method for reading images.
            # although you could just convert a PIL image to
            # a numpy array, too.
            img = cv2.imread(img_name)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Albumentations works more nicely if you
            augmented = self.aug(image=img)
            img = augmented["image"]

        else:

            img = Image.open(img_name).convert('RGB')
            
            # Apply resize 
            if self.transform:
                img = self.transform(img)


        return img.type(torch.FloatTensor), torch.tensor(img_class, dtype=torch.int64)