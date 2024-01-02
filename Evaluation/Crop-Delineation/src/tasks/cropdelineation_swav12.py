"""Field/crop delineation"""
from PIL import Image
import numpy as np
import cv2

import torch
import rasterio

from torch.utils.data import Dataset
import torchvision.datasets as datasets
import torchvision.transforms as transforms


def rasterio_loader(path: str) -> "np.typing.NDArray[np.int_]":

    with rasterio.open(path) as f:
        array: "np.typing.NDArray[np.int_]" = f.read().astype(np.int32)
        array = array.transpose(1, 2, 0)
        newarray = array[:, :, :10]  # / 10000
        newarr = np.concatenate((newarray, array[:, :, 10:]), axis=2)
    return newarr


class CropDelineationDataset12(datasets.ImageFolder):
    # class CropDelineationDataset12(datasets.ImageFolder):
    def __init__(self, path, files, mask_filled, transform=None, augmentations=None):
        super().__init__(root=path, loader=rasterio_loader)
        # self.root = path
        #  self.loader = rasterio_loader
        self.img_path = path  # Path to images
        self.files = files  # files in the image path
        self.mask_filled = mask_filled  # Path masks
        if mask_filled:
            self.mask_path = "/home/mh613/ssrs/crop_delineation/" + "masks_filled/"
        else:
            self.mask_path = "/home/mh613/ssrs/crop_delineation/" + "masks/"
        self.transform = transform
        self.aug = augmentations

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # path, _ = np.array(self.samples[idx])
        image = rasterio_loader(
            self.img_path + "batch/" + str(self.files[idx]) + ".tif"
        )
        # image = rasterio_loader("/scratch/yc506/crop_delineation/batch/7563014.tif")
        # Load the image and mask
        img_name = image
        #    img_name = self.img_path + str(self.files[idx]) + ".tiff"

        mask_name = self.mask_path + str(self.files[idx]) + ".png"

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
            # have the mask as a 2D array
            mask = cv2.imread(mask_name, cv2.IMREAD_GRAYSCALE) / 255

            # mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
            # Albumentations works more nicely if you
            augmented = self.aug(image=img, mask=mask)
            img = augmented["image"]
            mask = augmented["mask"]

        else:
            #   path, _ = np.array(self.samples[index])
            # image = self.loader(path)
            ff = transforms.ToTensor()
            # img = Image.open(img_name).convert("RGB")
            img = ff(img_name)
            # Apply transforms
            # if self.transform:
            #     img = self.transform(img)

            mask = np.array(Image.open(mask_name)) / 255
            mask = torch.from_numpy(mask)

        return img.type(torch.FloatTensor), mask.type(torch.LongTensor)
