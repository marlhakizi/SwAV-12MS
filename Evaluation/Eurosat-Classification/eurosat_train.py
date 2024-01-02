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


# Create parser
parser = argparse.ArgumentParser(
    description="""This script loads an encoder, a decoder, and a
    task, then trains using the specified set up on that task."""
)
###################
# Data args
###################
parser.add_argument(
    "--batch_size",
    type=int,
    default=16,
    help="""Batch size for the data. Default is 16, as this
    was found to work optimally on the GPUs we experiment with.""",
)
parser.add_argument(
    "--normalization",
    type=str,
    default="data",
    choices=["data", "dataTwelve"],
    help="""This specifies the normalization scheme to use
    when transforming the data for the model. Default is data-specific, but if you are using an
    ImageNet pretrained model, then specify 'imagenet' for quicker convergence.""",
)
parser.add_argument(
    "--data_size",
    type=int,
    default=None,
    help="""Some datasets allow you to specify 'small' so that you have a data limited
    experiment.""",
)
parser.add_argument(
    "--name",
    type=str,
    default=None,
    help="""Mentions the name of the type of model encoder it is based ons""",
)
parser.add_argument(
    "--nth",
    type=int,
    default=None,
    help="""Number of times to run script""",
)
###################
# Encoder arguments
###################
parser.add_argument(
    "--encoder",
    type=str,
    default="swav",
    choices=[
        "swav",
        "none",
        "imagenet",
        "swav-12",
        "swav-b3",
        "noweights",
    ],
    help="""The encoder to use. Valid
    options include 'swav', 'none', or 'imagenet'. If you specify, 'swav', then the encoder will
    load the pretrained model using the SwAV self-supervised method on ImageNet. 'none' loads
    a ResNet-50 with no pretrained weights (i.e., random weights). 'imagenet' loads the
    supervised pretrained model on ImageNet.""",
)

# Fine tuning for encoder
parser.add_argument(
    "--fine_tune_encoder",
    type=bool,
    default=False,
    help="""Whether to fine tune the encoder during
    supervision. If False, then gradients will not be calculated on the encoder. If True, then
    the gradients will be calculated. This prolongs training time by a little more than a minute
    per epoch.""",
)


###################
# Training arguments
###################
parser.add_argument(
    "--lr", type=float, default=1e-3, help="The learning rate. Default 1e-3."
)
parser.add_argument(
    "--weight_decay",
    type=float,
    default=0.0,
    help="Weight decay for parameters. Default 0.",
)
parser.add_argument(
    "--device",
    type=str,
    default="auto",
    help="""Whether to use the GPU. Default 'auto' which uses
    a GPU if they are available. It is recommended that you explicitly set the GPU using a value of
    'cuda:0' or 'cuda:1' so that you can more easily track the model.""",
)

parser.add_argument(
    "--epochs", type=int, default=100, help="Number of epochs. Default 100."
)
parser.add_argument(
    "--augment",
    type=bool,
    default=False,
    help="""Whether to apply advanced
    augmentations. By default, this is False. However, this should almost always be turned to
    True for future experiments to prevent overfitting and to increase accuracy.""",
)

###################
# Logging arguments
###################
parser.add_argument(
    "--dump_path",
    type=str,
    default="./experiments/results/",
    help="Where to put the results for analysis.",
    required=True,
)
parser.add_argument(
    "--log_level",
    type=str,
    default="INFO",
    help="The log level to report on. Default 'INFO'",
)


def main():
    global args
    args = parser.parse_args()
    validate_args()
    overall_timer = utils.Timer()
    try:
        final_dump_path = (
            args.dump_path
            + str(args.data_size)
            + "/"
            + args.name
            + "/"
            + str(args.nth)
            + "/"
        )

        os.makedirs(final_dump_path)

    except FileExistsError:
        print("Please delete the target directory if you would like to proceed.")
        return
    # define augmenation: example, random rotation, flip and transpos

    # Set up logger and log the arguments
    set_up_logger()
    logging.info(args)

    # Load the train dataset and the test dataset

    train_data, test_data = datasets._load_eurosat_data(
        args.normalization, args.augment, args.data_size
    )
    # Create the dataloader
    logging.info("Creating data loaders...")
    train_loader = DataLoader(train_data, batch_size=20, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=20, shuffle=False)
    # Instantiate the model
    logging.info("Instantiating the model...")
    encoder = encoders.load(args.encoder)
    decoder = RegLog(10)
    # encoder = encoders.load("swav-12")
    # # encoder=encoders.load("swav-b3")
    # decoder = RegLog(10)
    global DEVICE
    DEVICE = set_device("auto")
    # DEVICE = set_device('cpu')
    print("Device is " + DEVICE)
    encoder = encoder.to(DEVICE)
    decoder = decoder.to(DEVICE)
    save_path = (
        args.dump_path
        + str(args.data_size)
        + "/"
        + args.name
        + "/"
        + str(args.nth)
        + "/"
    )
    # probably needs to change
    optimizer = torch.optim.Adam(
        decoder.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    criterion = nn.CrossEntropyLoss().to(DEVICE)
    monitor_pth = "/home/mh613/climate-eye/dd"  # "/home/yc506/ssrs/experiments/results/eurosat_resize_swav"
    epoch_timer = utils.Timer()
    monitor = utils.PerformanceMonitor(monitor_pth)
    best_test_loss = float("inf")
    # do 100 epochs
    for epoch in range(args.epochs):
        print(f"Beginning epoch {epoch}")
        logging.info(f"Beginning epoch {epoch}...")

        loss = train(train_loader, encoder, decoder, optimizer, criterion)
        monitor.log(epoch, "train", loss)

        loss = test(test_loader, encoder, decoder, criterion)
        monitor.log(epoch, "val", loss)
        logging.info(f"Epoch {epoch} took {epoch_timer.minutes_elapsed()} minutes.")
        epoch_timer.reset()

        if loss < best_test_loss:
            logging.info("Saving model")
            save_model(encoder, decoder, monitor_pth, "best.pt")
            best_test_loss = loss
    save_model(encoder, decoder, save_path, "final.pt")
    logging.info(f"Code completed in {overall_timer.minutes_elapsed()}.")


# params = decoder.parameters()


def set_device(d):
    if d == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = d
    return device


def save_model(enc, dec, dump_path, name):
    torch.save(enc.state_dict(), os.path.join(dump_path, "enc_" + name))
    torch.save(dec.state_dict(), os.path.join(dump_path, "dec_" + name))


def train(loader, encoder, decoder, optimizer, criterion, fine_tune_encoder=False):

    if fine_tune_encoder:
        encoder.train()
    else:
        encoder.eval()

    decoder.train()
    criterion = criterion.to(DEVICE)
    avg_loss = utils.AverageMeter()
    num_batches = len(loader)
    for batch_idx, (inp, target) in enumerate(loader):
        if batch_idx % 10 == 0:
            print(f"Beginning batch {batch_idx} of {num_batches}")
        logging.debug(f"Training batch {batch_idx}...")
        # Move to the GPU
        inp = inp.to(DEVICE)
        target = target.to(DEVICE)

        if fine_tune_encoder:
            output = encoder(inp)
        else:
            with torch.no_grad():
                output = encoder(inp)
        ################### The change I'm not super sure #####################
        output = decoder(output[0])
        ########################################################################
        loss = criterion(output, target)

        if batch_idx % 10 == 0:
            print(f"\t Train Loss: {loss.item()}")
        # Calculate the gradients
        optimizer.zero_grad()
        loss.backward()
        avg_loss.update(loss.item(), inp.size(0))
        # Step forward
        optimizer.step()

    return avg_loss.avg


@torch.no_grad()
def test(data_loader, encoder, decoder, criterion):

    encoder.eval()
    decoder.eval()
    criterion = criterion.to(DEVICE)
    avg_loss = utils.AverageMeter()
    for batch_idx, (inp, target) in enumerate(data_loader):
        # Move to the GPU
        if batch_idx % 100 == 0:
            print(f"Testing batch {batch_idx}")
        inp = inp.to(DEVICE)
        target = target.to(DEVICE)

        # Compute output

        ################### The change I'm not super sure #####################
        output = encoder(inp)
        output = decoder(output[0])
        ########################################################################
        loss = criterion(output, target)
        avg_loss.update(loss.item(), inp.size(0))
        if batch_idx % 10 == 0:
            print(f"\t Test Loss: {loss.item()}")

    return avg_loss.avg


def validate_args():
    """
    This function ensures that several criteria
    are met before proceeding.
    """

    if args.encoder is None:
        raise Exception("An encoder must be specified.")


def set_up_logger():
    logging.basicConfig(
        filename=os.path.join(args.dump_path, "output.log"),
        filemode="w",
        level=args.log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


if __name__ == "__main__":
    main()
