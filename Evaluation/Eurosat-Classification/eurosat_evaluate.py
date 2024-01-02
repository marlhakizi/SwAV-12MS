from glob import glob
import argparse
from tqdm import tqdm
from itertools import chain

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from PIL import Image
import albumentations as A

from models import encoders, decoders
from src import datasets, utils, metrics
from torch import nn
from torch.nn import functional as F
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
from src.metrics import RegLog

bestOrFinal = "final"
weights_folder = "./dr/500/Eurosat_12/1"  # "results/field_delineation_10ep/" #
encoderWeights_path = os.path.join(weights_folder, f"enc_{bestOrFinal}.pt")
decoderWeights_path = os.path.join(weights_folder, f"dec_{bestOrFinal}.pt")
test_data = datasets._load_eurosat_data(
    normalization="dataTwelve", augmentations=False, size=500, evaluate=True
)
test_loader = DataLoader(test_data, batch_size=20, shuffle=False)
encoder_trained = encoder = encoders.load("swav-12")
decoder_trained = RegLog(10)
decoder_trained.load_state_dict(torch.load(decoderWeights_path))
encoder_trained = encoder_trained.eval()
decoder_trained = decoder_trained.eval()
dataloader = test_loader
global DEVICE


def set_device(d):
    if d == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = d
    return device


DEVICE = set_device("auto")
encoder = encoder_trained.to(DEVICE)
decoder = decoder_trained.to(DEVICE)


def get_prediction_accuracy(dataloader, encoder, decoder):
    # preds = []
    # targets = []
    correct = 0
    total = 0
    for i, (img, label) in enumerate(dataloader):
        # Load through the model.
        img = img.to(DEVICE)
        label = label.to(DEVICE)
        with torch.no_grad():
            output = encoder(img)
            output = decoder(output[0])
            pred_prob = torch.sigmoid(output)  # activation function sigmoid function
            _, predicted = torch.max(
                pred_prob, 1
            )  # the maximum probability class will be picked as our prediction
            total += label.size(0)  # the total number of prediction
            correct += (predicted == label).sum().item()
    return correct / total


print(get_prediction_accuracy(test_loader, encoder, decoder))
