from PIL import Image
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms

# import torchvision.datasets as datasets
import albumentations as A

from torch import nn
from torch.nn import functional as F
import rasterio


def rasterio_loader(path: str) -> "np.typing.NDArray[np.float_]":

    with rasterio.open(path) as f:
        array: "np.typing.NDArray[np.float_]" = f.read().astype(np.float32)
        array = array.transpose(1, 2, 0)
        newarray = array[:, :, :10]  # / 10000
        newarr = np.concatenate((newarray, array[:, :, 10:]), axis=2)
    return newarr


# class Eurosat12(datasets.ImageFolder):
class Eurosat12(Dataset):
    # """
    # This class takes in the folder with RGB Eurosat data and the file names after train/test split
    # Return the image tensor and the associated class of that image
    # """

    # def __init__(self, path, files, transform=None, augmentations=None):
    #     self.img_path = path  # Path to images
    #     self.files = files  # list of filenames in the file path
    #     self.transform = transforms.Compose(
    #         [
    #             transform,
    #             # this transformation here is to make sure we have
    #             transforms.CenterCrop(size=64),
    #             transforms.Resize(size=224),
    #         ]
    #     )
    #     self.aug = augmentations
    #     self.root = None  # add a root attribute

    # def __len__(self):
    #     return len(self.files)

    # def __getitem__(self, idx):

    #     # Load the image
    #     img_name = self.img_path + str(
    #         self.files[idx]
    #     )  # this will be something like '/scratch/yc506/eurosat_rgb/Residential_2054.tif'
    #     # extract the image class with the name
    #     name_class = (
    #         str(self.files[idx]).lower().split("_")[0]
    #     )  # this will give us something like 'forest'
    #     class_map = {
    #         "annualcrop": 0,
    #         "forest": 1,
    #         "herbaceousvegetation": 2,
    #         "highway": 3,
    #         "industrial": 4,
    #         "pasture": 5,
    #         "permanentcrop": 6,
    #         "residential": 7,
    #         "river": 8,
    #         "sealake": 9,
    #     }
    #     img_class = class_map[name_class]  # assign numerical class to each image

    #     # Apply albumentations augmentations
    #     # before applying the standard transform
    #     if self.aug:
    #         # Must convert to numpy array for
    #         # this to work nicely with albumentations
    #         # cv2 is the recommended method for reading images.
    #         # although you could just convert a PIL image to
    #         # a numpy array, too.
    #         img = cv2.imread(img_name)
    #         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    #         # Albumentations works more nicely if you
    #         augmented = self.aug(image=img)
    #         img = augmented["image"]

    #     else:

    #         img = Image.open(img_name).convert("RGB")

    #         # Apply resize
    #         if self.transform:
    #             img = self.transform(img)

    #     return img.type(torch.FloatTensor), torch.tensor(img_class, dtype=torch.int64)

    """
    This class takes in the folder with 12-channel Eurosat data and the file names after train/test split
    Return the image tensor and the associated class of that image
    """

    def __init__(self, path, files, transform=None, augmentations=None):
        # super().__init__(root=path, loader=rasterio_loader)
        self.img_path = path  # Path to images
        self.files = files  # list of filenames in the file path
        # self.transform = transform
        trans = []
        trans.extend(
            [
                transforms.Compose(
                    [
                        #  transform,
                        transforms.CenterCrop(size=64),
                        transforms.Resize(size=224),
                        transform
                        # transforms.ToTensor(),
                    ]
                )
            ]
        )
        self.trans = trans
        self.transform = transforms.Compose(
            [
                transform,
                # this transformation here is to make sure we have
                transforms.CenterCrop(size=64),
                transforms.Resize(size=224),
            ]
        )
        self.aug = augmentations

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        # Load the image
        img_name = self.img_path + str(
            self.files[idx]
        )  # this will be something like '/scratch/yc506/eurosat_rgb/Residential_2054.tif'
        img_name = rasterio_loader(img_name)

        # img_name = image
        # extract the image class with the name
        name_class = (
            str(self.files[idx]).lower().split("_")[0]
        )  # this will give us something like 'forest'
        class_map = {
            "annualcrop": 0,
            "forest": 1,
            "herbaceousvegetation": 2,
            "highway": 3,
            "industrial": 4,
            "pasture": 5,
            "permanentcrop": 6,
            "residential": 7,
            "river": 8,
            "sealake": 9,
        }
        img_class = class_map[name_class]  # assign numerical class to each image

        # Apply albumentations augmentations
        # before applying the standard transform

        #  print("ff")
        # img = Image.open(img_name).convert("RGB")
        #  img = rasterio_loader(img)
        ff = transforms.ToTensor()
        # img = ff(img_name)

        # img = ff(img_name)
        # Apply resize
        multi_crops = list(map(lambda trans: trans(ff(img_name)), self.trans))

        img = multi_crops[0]

        return img.type(torch.FloatTensor), torch.tensor(img_class, dtype=torch.int64)


# normalize = {
#     "mean": [
#         0.04560511,
#         0.07832425,
#         0.07050303,
#         0.13134229,
#         0.29649284,
#         0.35478109,
#         0.37182458,
#         0.38193435,
#         0.22069953,
#         0.13402856,
#         -12.5696884,
#         -18.2610512,
#     ],
#     "std": [
#         0.06170507,
#         0.06207748,
#         0.07390513,
#         0.06927535,
#         0.0757848,
#         0.09344653,
#         0.09964956,
#         0.09542389,
#         0.07652418,
#         0.0762175,
#         4.31484634,
#         3.4123834,
#     ],
# }
################ to be changed for eurosat 12-channel values #############


# tr_normalize = transforms.Normalize(mean=normalize["mean"], std=normalize["std"])
# train_transform = transforms.Compose([transforms.ToTensor(), tr_normalize])
# path = "/scratch/mh613/NewEuroSat/"
# train_files = ["SeaLake_99.tif", "SeaLake_99.tif"]


# # Create the train and test datasets using Eurosat12 class
# train_data = Eurosat12(path, train_files, transform=train_transform)


# print(Eurosat12(path, train_files, transform=train_transform))
# Define the path to your image folder and file names after train/test split

# normalize = {
#     "mean": [
#         0.04560511,
#         0.07832425,
#         0.07050303,
#         0.13134229,
#         0.29649284,
#         0.35478109,
#         0.37182458,
#         0.38193435,
#         0.22069953,
#         0.13402856,
#         -12.5696884,
#         -18.2610512,
#     ],
#     "std": [
#         0.06170507,
#         0.06207748,
#         0.07390513,
#         0.06927535,
#         0.0757848,
#         0.09344653,
#         0.09964956,
#         0.09542389,
#         0.07652418,
#         0.0762175,
#         4.31484634,
#         3.4123834,
#     ],
# }
# ################ to be changed for eurosat 12-channel values #############


# tr_normalize = transforms.Normalize(mean=normalize["mean"], std=normalize["std"])
# train_transform = transforms.Compose([tr_normalize])
# path = "/scratch/mh613/NewEuroSat/"
# train_files = ["SeaLake_99.tif", "SeaLake_99.tif"]


# # Create the train and test datasets using Eurosat12 class
# train_data = Eurosat12(path, train_files, transform=train_transform)
# train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
# for batch_idx, (data, target) in enumerate(train_loader):

#     assert (
#         data.shape[0] == 1
#     ), f"Batch {batch_idx} has size {data.shape[0]}, expected {batch_size}"
