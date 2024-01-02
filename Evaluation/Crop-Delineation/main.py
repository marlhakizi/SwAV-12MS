import os
import argparse
import logging
from itertools import chain

import torch
from torch.utils.data import DataLoader

from src import datasets, utils, metrics
from models import encoders, decoders
import wandb
import numpy as np
import torchvision.transforms as transforms

# Create parser
parser = argparse.ArgumentParser(
    description="""This script loads an encoder, a decoder, and a
    task, then trains using the specified set up on that task."""
)

###################
# Data args
###################
parser.add_argument(
    "--task",
    type=str,
    default=None,
    choices=["solar", "building", "crop_delineation"],
    required=True,
    help="""The training task to attempt. Valid tasks
    include 'solar' [todo -- add the list of tasks as they are developed].""",
)
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
    choices=["data", "imagenet", "1mRGB", "dataTwelve"],
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
# Decoder arguments
###################
parser.add_argument(
    "--decoder",
    type=str,
    default="unet",
    choices=["unet"],
    help="""The decoder to use. By default
    the decoder is 'unet' and no other methods are supported at this time.""",
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
    "--criterion",
    type=str,
    default="softiou",
    choices=["softiou", "xent"],
    help="""Select the criterion to use. By default, the
    criterion is 'softiou' (stylized: SoftIoU), and this should be the default value for semantic
    segmentation tasks, although 'xent' maps to Binary Cross Entropy Loss.""",
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
    # Set up arguments
    global args
    args = parser.parse_args()
    validate_args()

    # Set up timer to time results
    overall_timer = utils.Timer()
    # Set up path for recording results
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
    logging.info("Loading dataset...")
    train_data, test_data = datasets.load(
        args.task, args.normalization, args.augment, args.data_size
    )

    # Create the dataloader
    logging.info("Creating data loaders...")
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    # Instantiate the model
    logging.info("Instantiating the model...")
    encoder = encoders.load(args.encoder)
    decoder = decoders.load(args.decoder, encoder)

    # Load model to GPU
    logging.info("Loading model to device...")
    global DEVICE
    DEVICE = set_device(args.device)
    print("Device is " + DEVICE)
    encoder = encoder.to(DEVICE)
    decoder = decoder.to(DEVICE)

    config_dict = {
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
    }
    wandb.init(project="Crop-Del-RGB", config=config_dict)
    wandb.watch(decoder, log="all")
    save_path = (
        args.dump_path
        + str(args.data_size)
        + "/"
        + args.name
        + "/"
        + str(args.nth)
        + "/"
    )
    # Set up optimizer, depending on whether
    # we are fine-tuning or not
    if args.fine_tune_encoder:
        # Chain the iterators to combine them.
        params = chain(encoder.parameters(), decoder.parameters())
    else:
        params = decoder.parameters()

    logging.info("Setting up optimizer and criterion...")
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    criterion = metrics.load(args.criterion, DEVICE)

    epoch_timer = utils.Timer()
    monitor = utils.PerformanceMonitor(args.dump_path)
    best_test_loss = float("inf")
    for epoch in range(args.epochs):
        print(f"Beginning epoch {epoch}")
        logging.info(f"Beginning epoch {epoch}...")

        loss = train(train_loader, encoder, decoder, optimizer, criterion)
        monitor.log(epoch, "train", loss)
        wandb.log({"train loss": loss})
        loss = test(test_loader, encoder, decoder, criterion)
        monitor.log(epoch, "val", loss)
        logging.info(f"Epoch {epoch} took {epoch_timer.minutes_elapsed()} minutes.")
        epoch_timer.reset()
        wandb.log({"val loss": loss})
        if loss < best_test_loss:
            logging.info("Saving model")
            save_model(encoder, decoder, save_path, "best.pt")
            best_test_loss = loss
            print(loss)

    save_model(encoder, decoder, save_path, "final.pt")
    logging.info(f"Code completed in {overall_timer.minutes_elapsed()}.")

    # ## 1 define transform: normalization and augmenation
    # based on the benchmarkdataset use corresponding
    # comptuted mean and std,
    if args.normalization == "imagenet":
        mean = [0.2384, 0.2967, 0.3172]
        std = [0.1873, 0.1226, 0.1138]

        # transform_norm = transforms.Compose(
        #     [transforms.ToTensor(), transforms.Normalize(mean, std)]
        # )
    # TODO on NEW DATA
    if args.normalization == "data":
        mean = [0.24, 0.297, 0.317]
        std = [0.187, 0.123, 0.114]
    # invert transfomration when plotting

    if args.normalization == "dataTwelve":
        mean = [
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
        ]
        std = [
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
        ]
    invTrans = transforms.Compose(
        [
            transforms.Normalize(
                mean=[-mean[i] / std[i] for i in range(3)],
                std=[1 / std[i] for i in range(3)],
            ),
        ]
    )
    # log results to WanDB
    if args.normalization != "dataTwelve":
        table_train = makewandb_table_data(train_data, encoder, decoder, invTrans)
        table_test = makewandb_table_data(test_data, encoder, decoder, invTrans)
        wandb.log({"train_prediction": table_train})
        wandb.log({"test_prediction": table_test})


def save_model(enc, dec, dump_path, name):
    torch.save(enc.state_dict(), os.path.join(dump_path, "enc_" + name))
    torch.save(dec.state_dict(), os.path.join(dump_path, "dec_" + name))


def train(loader, encoder, decoder, optimizer, criterion):

    if args.fine_tune_encoder:
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

        if args.fine_tune_encoder:
            output = encoder(inp)
        else:
            with torch.no_grad():
                output = encoder(inp)

        output = decoder(output)
        # print(criterion.get_device())
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
        output = decoder(encoder(inp))
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

    if args.task is None:
        raise Exception("A task must be specified.")
    if args.encoder is None:
        raise Exception("An encoder must be specified.")
    if args.decoder is None:
        raise Exception("A decoder must be specified.")


def set_up_logger():
    logging.basicConfig(
        filename=os.path.join(args.dump_path, "output.log"),
        filemode="w",
        level=args.log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def set_device(d):
    if d == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = d
    return device


def img_np2torch_dim(X):
    X_ = np.vstack([np.expand_dims(X[:, :, i], 0) for i in range(3)])
    return X_


def img_torch2np(X):
    X_ = X.permute((1, 2, 0))
    X_ = X_.numpy()
    return X_


def makewandb_table_data(dataset, encoder, decoder, invTrans):
    # add prediction to wandb table
    wandb_table_data = []
    encoder.eval()
    decoder.to(DEVICE)
    np.random.seed(123)
    for i in np.random.choice(range(len(dataset)), 10, replace=False):
        X0, y = dataset[i]
        X = torch.unsqueeze(X0, 0)
        X = X.to(DEVICE)
        X_ = img_torch2np(invTrans(X0))
        y_ = img_torch2np(torch.unsqueeze(y, 0))
        pred = decoder(encoder(X))
        pred_prob = torch.sigmoid(pred)
        pred_np = img_torch2np(pred_prob.detach().cpu().squeeze(0))
        images_wandb = wandb.Image(X_)
        mask_wandb = wandb.Image(y_)
        predMask_wandb = wandb.Image(pred_np)
        wandb_table_data.append([images_wandb, mask_wandb, predMask_wandb])
        columns = ["image", "truth", "prediction"]
        wandb_table = wandb.Table(data=wandb_table_data, columns=columns)
    return wandb_table


if __name__ == "__main__":
    main()
