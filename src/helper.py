import sys
import os
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

sys.path.append("src/")

from config import PROCESSED_DATA_PATH
from utils import load, params

from discriminator import Discriminator
from generator import Generator
from feature_extractor import VGG16


def load_dataloader():
    """
    Loads training and testing dataloaders from the processed data path.

    Raises:
        Exception: If the processed data path does not exist or dataloaders cannot be found.

    Returns:
        tuple: A tuple containing the train and test dataloaders.
    """
    if os.path.exists(PROCESSED_DATA_PATH):
        train_dataloader = load(
            filename=os.path.join(PROCESSED_DATA_PATH, "train_dataloader.pkl")
        )
        test_dataloader = load(
            filename=os.path.join(PROCESSED_DATA_PATH, "test_dataloader.pkl")
        )
    else:
        raise Exception("Dataset not found".capitalize())

    return train_dataloader, test_dataloader


def helper(**kwargs):
    """
    Initializes the GAN components including models, optimizers, and loss functions based on the provided configurations.

    Parameters:
        kwargs (dict): A dictionary containing configuration options such as learning rate, optimizers, and device.

    Returns:
        dict: A dictionary containing initialized components including models, dataloaders, optimizers, schedulers, and loss functions.
    """
    lr = kwargs["lr"]
    beta1 = kwargs["beta1"]
    adam = kwargs["adam"]
    SGD = kwargs["SDG"]
    device = kwargs["device"]
    lr_scheduler = kwargs["is_lr_scheduler"]

    try:
        train_dataloader, test_dataloader = load_dataloader()

    except Exception as e:
        print("The exception is: ", e)

    if adam:
        try:
            netG = Generator().to(device)
            netD = Discriminator().to(device)

        except Exception as e:
            print("The exception caught in the section # {}".format(e).capitalize())
        else:

            optimizerG = optim.Adam(
                netG.parameters(), lr=lr, betas=(beta1, params()["helpers"]["beta2"])
            )
            optimizerD = optim.Adam(
                netD.parameters(), lr=lr, betas=(beta1, params()["helpers"]["beta2"])
            )

    elif SGD:
        try:
            netG = Generator().to(device)
            netD = Discriminator().to(device)

        except Exception as e:
            print("The exception caught in the section # {}".format(e).capitalize())
        else:
            optimizerG = optim.SGD(netG.parameters(), lr=lr, momentum=beta1)
            optimizerD = optim.SGD(netD.parameters(), lr=lr, momentum=beta1)

    if lr_scheduler:
        schedulerG = StepLR(
            optimizerG,
            step_size=params()["helpers"]["lr_steps"],
            gamma=params()["helpers"]["lr_gamma"],
        )
        schedulerD = StepLR(
            optimizerD,
            step_size=params()["helpers"]["lr_steps"],
            gamma=params()["helpers"]["lr_gamma"],
        )

    try:
        content_loss = VGG16(pretrained=True).to(device)
        adversarial_loss = nn.MSELoss()

    except Exception as e:
        print("The exception caught in the section # {}".format(e).capitalize())

    return {
        "train_dataloader": train_dataloader,
        "test_dataloader": test_dataloader,
        "netG": netG,
        "netD": netD,
        "optimizerG": optimizerG,
        "optimizerD": optimizerD,
        "schedulerG": schedulerG,
        "schedulerD": schedulerD,
        "adversarial_loss": adversarial_loss,
        "content_loss": content_loss,
    }


if __name__ == "__main__":
    init = helper(
        lr=2e-4,
        beta1=0.5,
        adam=True,
        SGD=False,
        device="mps",
        is_lr_scheduler=False,
        display=True,
    )
