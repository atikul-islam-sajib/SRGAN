import sys
import argparse
import torch
import torch.nn as nn

sys.path.append("src/")

from helpers import helper
from utils import weight_int


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

                self.netG = self.netG.apply(weight_int)
                self.netD = self.netD.apply(weight_int)

            self.optimizerG = init["optimizerG"]
            self.optimizerD = init["optimizerD"]
            self.schedulerG = init["schedulerG"]
            self.schedulerD = init["schedulerD"]
            self.adversarial_loss = init["adversarial_loss"]
            self.content_loss = init["content_loss"]

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

    def save_checkpoints(self):
        pass

    def update_discriminator_training(self):
        pass

    def update_generator_training(self):
        pass

    def save_training_images(self):
        pass

    def show_progress(self):
        pass

    def train(self):
        pass

    @staticmethod
    def plot_history():
        pass


if __name__ == "__main__":
    trainer = Trainer(
        epochs=1,
        lr=1e-4,
        device="mps",
        adam=True,
        SGD=False,
        beta1=0.5,
        is_lr_scheduler=True,
    )
    trainer.train()

    print(trainer.netD)
