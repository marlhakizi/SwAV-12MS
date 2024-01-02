import os
import logging
import numpy as np
import pandas as pd
import joblib
import random
from torchvision import transforms
from natsort import natsorted
import albumentations as A
from albumentations.pytorch import ToTensorV2

from sklearn.model_selection import train_test_split

from .tasks.solar import SolarPVDataset
from .tasks.building import BuildingSegmentationDataset
from .tasks.cropdelineation import CropDelineationDataset
from .tasks.eurosat import Eurosat
from .tasks.eurosat_swav12 import Eurosat12
from .tasks.cropdelineation_swav12 import CropDelineationDataset12


def load(
    task,
    normalization="data",
    augmentations=False,
    data_size=1095,
    evaluate=False,
    old=False,
):
    logging.debug(f"In datasets, the task {task} is being loaded.")

    if task == "crop_delineation":

        print("Loading crop delineation dataset.")
        return _load_cropdel_data(normalization, augmentations, evaluate, data_size)
    elif task == "eurosat":
        print("Loading Eurosat dataset.")
        return _load_eurosat_data(normalization, augmentations, evaluate, data_size)


def _load_cropdel_data(normalization, augmentations, evaluate, size=None):
    print(f"Data evaluate: {evaluate}")
    """
    This function takes care of loading the crop segmentation
    data for training the model.
    """
    # random.seed(123)
    # Change this to false if you want to use a different set of masks
    mask_filled = False
    second_list = [
        int(i.split(".")[0])
        for i in os.listdir("/scratch/yc506/crop_delineation/batch")
    ]
    file_map = pd.read_csv("/home/mh613/ssrs/crop_delineation/" + "/clean_data.csv")
    if normalization != "dataTwelve":
        data_path = "/home/mh613/ssrs/crop_delineation/"
    else:
        data_path = "/scratch/yc506/crop_delineation/"
        # This loads the list of files to reference

        # if size is not None:
        #     print(f"Loading crop delineation training data with size {size}.")
        #     train_files = list(joblib.load(data_path + f"train_{size}.joblib"))
        # else:
    print("Loading the complete training dataset.")

    old_train_files = [
        i
        for i in list(file_map[file_map["split"] == "train"]["indices"])
        if i in second_list
    ]
    if size != 1095:

        train_files = random.sample(old_train_files, size)
    else:
        train_files = old_train_files
    val_files = [
        i
        for i in list(file_map[file_map["split"] == "val"]["indices"])
        if i in second_list
    ]
    test_files = [
        i
        for i in list(file_map[file_map["split"] == "test"]["indices"])
        if i in second_list
    ]
    if normalization == "data":
        # TODO -- calculate this
        normalize = {"mean": [0.24, 0.297, 0.317], "std": [0.187, 0.123, 0.114]}
    elif normalization == "imagenet":
        normalize = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
    elif normalization == "dataTwelve":
        normalize = {
            "mean": [
                0.04560511,
                0.07832425,
                0.07050303,
                0.13134229,
                0.29649284,
                0.35478109,
                0.37182458,
                0.38193435,
                0.22069953,
                0.13402856,
                -12.5696884,
                -18.2610512,
            ],
            "std": [
                0.06170507,
                0.06207748,
                0.07390513,
                0.06927535,
                0.0757848,
                0.09344653,
                0.09964956,
                0.09542389,
                0.07652418,
                0.0762175,
                4.31484634,
                3.4123834,
            ],
        }
    # Add augmentations
    if augmentations:
        print("Adding augmentations...")
        aug = A.Compose(
            [
                A.RandomRotate90(),
                A.VerticalFlip(),
                A.HorizontalFlip(),
                A.Transpose(),
                A.Normalize(mean=normalize["mean"], std=normalize["std"]),
                ToTensorV2(),
            ]
        )

    tr_normalize = transforms.Normalize(mean=normalize["mean"], std=normalize["std"])

    train_transform = transforms.Compose([transforms.ToTensor(), tr_normalize])

    test_transform = transforms.Compose([transforms.ToTensor(), tr_normalize])

    # Create the train dataset
    logging.debug("Creating the training dataset.")
    if augmentations:
        train_dataset = CropDelineationDataset(
            data_path,
            train_files,
            mask_filled,
            transform=train_transform,
            augmentations=aug,
        )
    else:
        if normalization == "dataTwelve":
            train_dataset = CropDelineationDataset12(
                data_path, train_files, mask_filled, transform=train_transform
            )
        else:
            train_dataset = CropDelineationDataset(
                data_path, train_files, mask_filled, transform=train_transform
            )
    # Load the test dataset
    logging.debug("Creating the test dataset.")
    if normalization == "dataTwelve":
        val_dataset = CropDelineationDataset12(
            data_path, val_files, mask_filled, transform=test_transform
        )
        test_dataset = CropDelineationDataset12(
            data_path, test_files, mask_filled, transform=test_transform
        )
    else:
        val_dataset = CropDelineationDataset(
            data_path, val_files, mask_filled, transform=test_transform
        )
        # Return the training and test dataset
        test_dataset = CropDelineationDataset(
            data_path, test_files, mask_filled, transform=test_transform
        )
    if evaluate:
        return test_dataset
    else:
        return train_dataset, val_dataset


"""
This function loads the data for Eurosat
"""


def _load_eurosat_data(normalization, evaluate, size=None, augmentations=False):

    if normalization == "dataTwelve":
        # 12-channel files
        data_path = "/scratch/mh613/NewEuroSat/"
    else:
        # rgb files
        data_path = "/scratch/yc506/NewEuroSat_rgb/"
    # a list of files in the data path
    file_set = os.listdir(data_path)

    # train val test split
    old_train_files, rem = train_test_split(file_set, test_size=0.2, random_state=123)
    # 11736 is the number of training file we will have 0.8 *14k
    if size != 11736:
        train_files = random.sample(old_train_files, size)
    else:
        train_files = old_train_files
    test_files, val_files = train_test_split(rem, test_size=0.5, random_state=123)

    if normalization != "dataTwelve":
        # mean and std for RGB data
        normalize = {
            "mean": [44.04389265, 49.10599284, 31.27095929],
            "std": [41.61620523, 38.21741166, 31.96225992],
        }
    elif normalization == "dataTwelve":
        ################ to be changed for eurosat 12-channel values #############
        normalize = {
            "mean": [
                616.95329649,
                867.83717358,
                842.86595966,
                1285.77942965,
                2308.69512439,
                2650.97561424,
                2791.82155065,
                2832.1399957,
                1967.90223554,
                1343.787036,
                -11.92402614,
                -18.43180948,
            ],
            "std": [
                850.36421999,
                837.83275801,
                931.51194456,
                920.84945907,
                1222.92903347,
                1426.90538092,
                1514.24879853,
                1491.36178338,
                1063.20240616,
                917.98727869,
                5.22952371,
                5.06522831,
            ],
        }

        ################ to be changed for eurosat 12-channel values #############
    tr_normalize = transforms.Normalize(mean=normalize["mean"], std=normalize["std"])
    train_transform = transforms.Compose([transforms.ToTensor(), tr_normalize])
    test_transform = transforms.Compose([transforms.ToTensor(), tr_normalize])
    print(augmentations)
    if augmentations:
        print("Adding augmentations...")
        aug = A.Compose(
            [
                A.RandomRotate90(),
                A.VerticalFlip(),
                A.HorizontalFlip(),
                A.Transpose(),
                A.Normalize(mean=normalize["mean"], std=normalize["std"]),
                ToTensorV2(),
            ]
        )
    logging.debug("Creating the training dataset.")
    if augmentations:
        train_dataset = Eurosat(
            data_path,
            train_files,
            transform=train_transform,
            augmentations=aug,
        )
    else:
        if normalization == "dataTwelve":
            train_dataset = Eurosat12(
                data_path, train_files, transform=transforms.Compose([tr_normalize])
            )
        else:
            train_dataset = Eurosat(data_path, train_files, transform=train_transform)

    logging.debug("Creating the test dataset.")
    if normalization == "dataTwelve":
        val_dataset = Eurosat12(
            data_path, val_files, transform=transforms.Compose([tr_normalize])
        )
        test_dataset = Eurosat12(
            data_path, test_files, transform=transforms.Compose([tr_normalize])
        )
    else:
        val_dataset = Eurosat(data_path, val_files, transform=test_transform)
        # Return the training and test dataset
        test_dataset = Eurosat(data_path, test_files, transform=test_transform)

    # train_dataset = Eurosat(data_path, train_files, transform=train_transform)
    #  val_dataset = Eurosat(data_path, val_files, transform=test_transform)

    # Return the training and test dataset
    # test_dataset = Eurosat(data_path, test_files, transform=test_transform)
    if evaluate:
        return test_dataset

    return train_dataset, val_dataset
