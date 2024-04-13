import sys
import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

import warnings

sys.path.append("src/")

from config import TRAIN_MODELS, BEST_MODELS, BEST_MODEL
from helpers import helper
from utils import weight_init

import warnings

warnings.filterwarnings("ignore")


class Trainer:
    """
    A class responsible for setting up and running training sessions for a GAN architecture,
    specifically for models dealing with image data. The trainer initializes models, dataloaders,
    optimizers, and loss functions, and handles training epochs.

    Attributes:
        epochs (int): Number of epochs to train the model.
        lr (float): Learning rate for the optimizer.
        device (str): Device to run the model on ('cuda', 'mps', or 'cpu').
        adam (bool): Flag to use Adam optimizer; mutually exclusive with SGD.
        SGD (bool): Flag to use SGD optimizer; mutually exclusive with Adam.
        beta1 (float): Beta1 hyperparameter for the Adam optimizer.
        is_l1 (bool): If True, includes L1 regularization in the loss calculation.
        is_l2 (bool): If True, includes L2 regularization in the loss calculation.
        is_elastic_net (bool): If True, includes both L1 and L2 regularizations as Elastic Net in the loss calculation.
        is_lr_scheduler (bool): If True, includes a learning rate scheduler.
        display (bool): If True, training progress and statistics will be displayed.
    """

    def __init__(
        self,
        epochs=100,
        lr=0.0002,
        device="mps",
        adam=True,
        SGD=False,
        beta1=0.5,
        is_l1=False,
        is_l2=False,
        is_elastic_net=False,
        is_lr_scheduler=False,
        display=True,
    ):
        """
        Initializes the Trainer object with the necessary configuration and parameters for training.
        Loads the data, models, optimizers, and loss functions.

        Parameters:
            epochs (int): Number of training epochs.
            lr (float): Learning rate for the optimizers.
            device (str): Device type for training ('cuda', 'mps', 'cpu').
            adam (bool): Whether to use Adam optimizer.
            SGD (bool): Whether to use SGD optimizer.
            beta1 (float): Beta1 parameter for the Adam optimizer.
            is_l1 (bool): Enables L1 regularization if True.
            is_l2 (bool): Enables L2 regularization if True.
            is_elastic_net (bool): Enables Elastic Net regularization (L1 + L2) if True.
            is_lr_scheduler (bool): If True, applies a learning rate scheduler.
            display (bool): If set to True, displays training progress.
        """
        self.epochs = epochs
        self.lr = lr
        self.device = device
        self.adam = adam
        self.SGD = SGD
        self.beta1 = beta1
        self.is_l1 = is_l1
        self.is_l2 = is_l2
        self.is_elastic_net = is_elastic_net
        self.is_lr_scheduler = is_lr_scheduler
        self.display = display

        try:
            init = helper(
                lr=self.lr,
                beta1=self.beta1,
                adam=self.adam,
                SGD=self.SGD,
                device=self.device,
                is_lr_scheduler=self.is_lr_scheduler,
            )
        except Exception as e:
            print("The exception caught in the section # {}".format(e).capitalize())

        else:
            self.train_dataloader = init["train_dataloader"]
            self.test_dataloader = init["test_dataloader"]

            try:
                self.netG = init["netG"]
                self.netD = init["netD"]

            except Exception as e:
                print("The exception caught in the section # {}".format(e).capitalize())

            finally:

                self.netG = self.netG.apply(weight_init)
                self.netD = self.netD.apply(weight_init)

            self.optimizerG = init["optimizerG"]
            self.optimizerD = init["optimizerD"]
            self.schedulerG = init["schedulerG"]
            self.schedulerD = init["schedulerD"]
            self.adversarial_loss = init["adversarial_loss"]
            self.content_loss = init["content_loss"]

            self.infinity = float("inf")
            self.loss_track = {"netG": [], "netD": []}

    def l1(self, model):
        """
        Calculates the L1 norm (sum of absolute values) of all the parameters in the given model.

        Parameters:
            model (torch.nn.Module): The model whose parameters' L1 norm is to be calculated.

        Returns:
            float: The L1 norm of the model's parameters.

        Raises:
            Exception: If no model is provided.
        """
        if model is not None:
            return (
                torch.norm(input=params, p=1) for params in model.parameters()
            ).sum()
        else:
            raise Exception("Model should be provided".capitalize())

    def l2(self, model):
        """
        Calculates the L2 norm (square root of the sum of squares) of all the parameters in the given model.

        Parameters:
            model (torch.nn.Module): The model whose parameters' L2 norm is to be calculated.

        Returns:
            float: The L2 norm of the model's parameters.

        Raises:
            Exception: If no model is provided.
        """
        if model is not None:
            return (
                torch.norm(input=params, p=2) for params in model.parameters()
            ).sum()
        else:
            raise Exception("Model should be provided".capitalize())

    def elastic_net(self, model):
        """
        Calculates the Elastic Net regularization, which is the sum of L1 and L2 norms, for all the parameters
        in the given model.

        Parameters:
            model (torch.nn.Module): The model whose parameters' Elastic Net regularization is to be calculated.

        Returns:
            float: The sum of L1 and L2 norms of the model's parameters.

        Raises:
            Exception: If no model is provided.
        """
        if model is not None:
            l1 = self.l1(model=model)
            l2 = self.l2(model=model)

            return l1 + l2
        else:
            raise Exception("Model should be provided".capitalize())

    def save_checkpoints(self, **kwargs):
        if (
            (os.path.exists(TRAIN_MODELS))
            and (os.path.exists(BEST_MODELS))
            and os.path.exists(BEST_MODEL)
        ):
            torch.save(
                {
                    "netG": self.netG.state_dict(),
                    "netG_loss": kwargs["netG_loss"],
                },
                os.path.join(TRAIN_MODELS, "netG{}.pth".format(kwargs["epoch"])),
            )

            self.loss_track["netG"].append(kwargs["netG_loss"])
            self.loss_track["netD"].append(kwargs["netD_loss"])

            if self.infinity > kwargs["netG_loss"]:

                self.infinity = kwargs["netG_loss"]

                torch.save(
                    {
                        "netG": self.netG.state_dict(),
                        "netG_loss": kwargs["netG_loss"],
                    },
                    os.path.join(BEST_MODELS, "netG{}.pth".format(kwargs["epoch"])),
                )

        else:
            raise FileExistsError("The directory should be created".capitalize())

    def update_discriminator_training(self, **kwargs):
        try:
            self.optimizerD.zero_grad()

            hr_loss = self.adversarial_loss(
                self.netD(kwargs["hr_images"]), kwargs["real_labels"]
            )
            fake_loss = self.adversarial_loss(
                self.netD(self.netG(kwargs["lr_images"])), kwargs["fake_labels"]
            )

            total_loss = 0.5 * (hr_loss + fake_loss)

            total_loss.backward()
            self.optimizerD.step()

        except KeyError as e:
            print("The exception caught in (Discriminator) # {}".format(e).capitalize())

        except Exception as e:
            print("The exception caught in (Discriminator)# {}".format(e).capitalize())

        else:
            return total_loss.item()

    def update_generator_training(self, **kwargs):
        try:
            self.optimizerD.zero_grad()

            generated_hr = self.netG(kwargs["lr_images"])

            adversarial_loss = self.adversarial_loss(
                generated_hr, kwargs["real_labels"]
            )

            real_features = self.content_loss(kwargs["hr_images"])
            fake_features = self.content_loss(generated_hr)

            content_loss = 1e-3 * torch.abs(real_features - fake_features).mean()

            total_loss = adversarial_loss + content_loss

            total_loss.backward()
            self.optimizerG.step()

        except KeyError as e:
            print("The exception caught in (Generator) # {}".format(e).capitalize())

        except Exception as e:
            print("The exception caught in (Generator)# {}".format(e).capitalize())

        else:
            return total_loss.item()

    def validate_model_on_test_data(self, **kwargs):
        try:
            generated_hr = self.netG(kwargs["lr_images"])

            loss = 0.5 * self.adversarial_loss(generated_hr, kwargs["hr_images"])

        except KeyError as e:
            print("The exception caught in # {}".format(e).capitalize())

        except Exception as e:
            print("The exception caught in # {}".format(e).capitalize())

        else:
            return loss.item()

    def save_training_images(self):
        pass

    def show_progress(self, **kwargs):
        pass

    def train(self):
        warnings.filterwarnings("ignore")

        for epoch in tqdm(range(self.epochs)):
            self.netG_loss = list()
            self.netD_loss = list()
            self.test_loss = list()

            for _, (lr_images, hr_images) in enumerate(self.train_dataloader):
                lr_images = lr_images.to(self.device)
                hr_images = hr_images.to(self.device)
                batch_size = hr_images.size(0)

                real_labels = torch.ones((batch_size,)).to(self.device)
                fake_labels = torch.zeros((batch_size,)).to(self.device)

                D_loss = self.update_discriminator_training(
                    lr_images=lr_images,
                    hr_images=hr_images,
                    real_labels=real_labels,
                    fake_labels=fake_labels,
                )
                G_loss = self.update_generator_training(
                    lr_images=lr_images, hr_images=hr_images, real_labels=real_labels
                )

                self.netG_loss.append(G_loss)
                self.netD_loss.append(D_loss)

            for _, (lr_images, hr_images) in enumerate(self.test_dataloader):
                lr_images = lr_images.to(self.device)
                hr_images = hr_images.to(self.device)

                loss = self.validate_model_on_test_data(
                    lr_images=lr_images, hr_images=hr_images
                )

                self.test_loss.append(loss)

            try:
                self.save_checkpoints(
                    epoch=epoch + 1,
                    netG_loss=np.array(self.netG_loss).mean(),
                    netD_loss=np.array(self.netD_loss).mean(),
                )
            except Exception as e:
                print("The exception caught in # {}".format(e).capitalize())

            print(
                "Epochs - [{}/{}] - train_netG_loss: {:.5f} - train_netD_loss: {:.5f} - test_loss: {:.5f}".format(
                    epoch + 1,
                    self.epochs,
                    np.mean(self.netG_loss),
                    np.mean(self.netD_loss),
                    np.mean(self.test_loss),
                )
            )

        # try:
        #     print(pd.DataFrame(self.loss_track))

        # except Exception as e:
        #     print("The exception caught in # {}".format(e).capitalize())

    @staticmethod
    def plot_history():
        pass


if __name__ == "__main__":
    trainer = Trainer(
        epochs=1,
        lr=0.0002,
        device="mps",
        adam=True,
        SGD=False,
        beta1=0.5,
    )
    trainer.train()
