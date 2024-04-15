import sys
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm
from torchvision.utils import save_image

import warnings

warnings.filterwarnings("ignore")

sys.path.append("src/")

from config import (
    TRAIN_MODELS,
    BEST_MODELS,
    BEST_MODEL,
    TRAIN_IMAGES,
    METRICS_PATH,
    MODEL_HISTORY,
)
from helpers import helper
from utils import weight_init, dump, load, clean

import warnings

warnings.filterwarnings("ignore")


class Trainer:
    """
    A class responsible for setting up and conducting training sessions for Generative Adversarial Networks (GANs).
    It handles the initialization of models, dataloaders, optimizers, and loss functions, orchestrating the training
    process and optionally applying various regularization strategies and learning rate schedules.

    Attributes:
        epochs (int): Total number of epochs to run the training.
        lr (float): Learning rate for the optimizers.
        content_loss (float): Coefficient for the content loss component.
        device (str): Device identifier to specify the computation device ('cuda', 'mps', 'cpu').
        adam (bool): If True, use the Adam optimizer; mutually exclusive with SGD.
        SGD (bool): If True, use the Stochastic Gradient Descent optimizer; mutually exclusive with Adam.
        beta1 (float): Beta1 hyperparameter for Adam optimizer.
        is_l1 (bool): Flag to enable L1 regularization.
        is_l2 (bool): Flag to enable L2 regularization.
        is_elastic_net (bool): Flag to enable Elastic Net regularization (combination of L1 and L2).
        is_lr_scheduler (bool): Flag to enable a learning rate scheduler.
        is_weight_init (bool): Flag to apply weight initialization to models.
        is_display (bool): Flag to control display of training progress and results.
        train_dataloader (DataLoader): DataLoader for training data.
        test_dataloader (DataLoader): DataLoader for test data.
        netG (nn.Module): Generator model.
        netD (nn.Module): Discriminator model.
        optimizerG (Optimizer): Optimizer for the generator.
        optimizerD (Optimizer): Optimizer for the discriminator.
        schedulerG (lr_scheduler): Learning rate scheduler for the generator.
        schedulerD (lr_scheduler): Learning rate scheduler for the discriminator.
        adversarial_loss (nn.Module): Loss function for the adversarial training.
        criterion_content (nn.Module): Loss function for content loss.
        infinity (float): A placeholder for tracking infinite loss or other unbounded metrics.
        loss_track (dict): A dictionary to keep track of loss values.
        history (dict): A dictionary to store historical data of metrics for analysis.
    """

    def __init__(
        self,
        epochs=100,
        lr=0.0002,
        content_loss=1e-2,
        device="mps",
        adam=True,
        SGD=False,
        beta1=0.5,
        is_l1=False,
        is_l2=False,
        is_elastic_net=False,
        is_lr_scheduler=False,
        is_weight_init=False,
        is_weight_clip=False,
        display=True,
    ):
        """
        Initializes the Trainer with configurations for training a GAN, setting up models,
        optimizers, and loss functions based on the provided parameters.

        Parameters:
            epochs (int): Number of epochs for training.
            lr (float): Initial learning rate for optimizers.
            content_loss (float): Multiplier for the content loss component.
            device (str): Target device for training ('cuda', 'mps', 'cpu').
            adam (bool): Whether to use the Adam optimizer.
            SGD (bool): Whether to use the Stochastic Gradient Descent optimizer.
            beta1 (float): The exponential decay rate for the first moment estimates in Adam.
            is_l1 (bool): Enable L1 regularization if True.
            is_l2 (bool): Enable L2 regularization if True.
            is_elastic_net (bool): Enable Elastic Net regularization (L1+L2) if True.
            is_lr_scheduler (bool): Activate a learning rate scheduler if True.
            is_weight_init (bool): Apply custom weight initialization to models if True.
            display (bool): Display training progress and statistics if True.
        """
        self.epochs = epochs
        self.lr = lr
        self.content_loss = content_loss
        self.device = device
        self.adam = adam
        self.SGD = SGD
        self.beta1 = beta1
        self.is_l1 = is_l1
        self.is_l2 = is_l2
        self.is_elastic_net = is_elastic_net
        self.is_lr_scheduler = is_lr_scheduler
        self.is_weight_init = is_weight_init
        self.is_weight_clip = is_weight_clip
        self.is_display = display

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

            if self.is_weight_init:
                self.netG = init["netG"].apply(weight_init)
                self.netD = init["netD"].apply(weight_init)

            else:
                self.netG = init["netG"]
                self.netD = init["netD"]

            self.optimizerG = init["optimizerG"]
            self.optimizerD = init["optimizerD"]
            self.schedulerG = init["schedulerG"]
            self.schedulerD = init["schedulerD"]
            self.adversarial_loss = init["adversarial_loss"]
            self.criterion_content = init["criterion_loss"]

            self.infinity = float("inf")
            self.clip_value = 0.01
            self.loss_track = {"netG": list(), "netD": list()}
            self.history = {"netG": list(), "netD": list()}

        finally:
            clean()

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
        """
        Saves the current state of the generator model as a checkpoint file during training. It stores both the model's
        state dictionary and the current loss. If the current loss is lower than previously recorded losses, it updates
        the best model checkpoint.

        Parameters:
            kwargs (dict): Contains various keyword arguments including:
                - epoch (int): The current training epoch, used to label the saved files.
                - netG_loss (float): The current loss of the generator.
                - netD_loss (float): The current loss of the discriminator, used for tracking only.

        Raises:
            FileExistsError: If the necessary directories for saving the models do not exist.

        Note:
            Checkpoints are saved in two directories: one for regular interval saves ('TRAIN_MODELS') and one for
            best model performance ('BEST_MODELS'). The function checks for the existence of these directories before
            attempting to save, raising an error if they are not found.
        """
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
        """
        Performs a training update on the Discriminator model. This method computes the adversarial loss for both
        real and generated (fake) images, combines these losses, and performs a backpropagation step to update
        the Discriminator's weights.

        Parameters:
            kwargs (dict): Keyword arguments containing necessary inputs for training:
                - hr_images (torch.Tensor): A batch of high-resolution (real) images.
                - lr_images (torch.Tensor): A batch of low-resolution images to be processed by the Generator to create fake samples.
                - real_labels (torch.Tensor): Ground truth labels for real images, typically ones.
                - fake_labels (torch.Tensor): Labels for fake images, typically zeros.

        Returns:
            float: The discriminator's total loss for the current training step, calculated as the average of the losses
            for real and fake images.

        Raises:
            KeyError: If any of the required keys ('hr_images', 'lr_images', 'real_labels', 'fake_labels') is missing in kwargs.
            Exception: General exceptions that could occur during loss calculation or backpropagation, providing error details.

        Notes:
            This method optimizes the discriminator by encouraging it to correctly distinguish between real and generated images.
            Loss is calculated separately for real and generated images, then averaged to form the total loss which is then backpropagated.
        """
        try:
            self.optimizerD.zero_grad()

            hr_loss = self.adversarial_loss(
                self.netD(kwargs["hr_images"]), kwargs["real_labels"]
            )
            fake_loss = self.adversarial_loss(
                self.netD(self.netG(kwargs["lr_images"])), kwargs["fake_labels"]
            )

            total_loss = 0.5 * (hr_loss + fake_loss)

            if self.is_l2:
                total_loss += self.l1(self.netD)

            if self.is_elastic_net:
                total_loss += self.elastic_net(self.netD)

            total_loss.backward()

            if self.is_weight_clip:
                for params in self.netD.parameters():
                    params.data.clamp_(-self.clip_value, self.clip_value)

            self.optimizerD.step()

        except KeyError as e:
            print("The exception caught in (Discriminator) # {}".format(e).capitalize())

        except Exception as e:
            print("The exception caught in (Discriminator)# {}".format(e).capitalize())

        else:
            return total_loss.item()

    def update_generator_training(self, **kwargs):
        """
        Conducts a training update for the Generator model. This method calculates the adversarial loss against
        the Discriminator's predictions and a content loss using VGG-based feature comparisons between the
        generated high-resolution images and real high-resolution images.

        Parameters:
            kwargs (dict): Keyword arguments containing the necessary inputs for training:
                - lr_images (torch.Tensor): A batch of low-resolution images that the Generator will transform.
                - hr_images (torch.Tensor): A batch of high-resolution images for reference.
                - real_labels (torch.Tensor): Ground truth labels for real images, typically ones, used for adversarial loss.

        Returns:
            float: The total loss for the generator at this training step, combining adversarial and content losses.

        Raises:
            KeyError: If any of the required keys ('lr_images', 'hr_images', 'real_labels') is missing in kwargs.
            Exception: Catches and logs other exceptions that may occur during loss calculation or optimization steps.

        Notes:
            The generator's total loss is a weighted sum of the adversarial loss, where the generator tries to fool the
            discriminator, and the content loss, which measures the perceptual difference between the generated images
            and the real images based on their feature representations extracted by a pretrained VGG network.
        """
        try:
            self.optimizerG.zero_grad()

            generated_hr = self.netG(kwargs["lr_images"])

            adversarial_loss = self.adversarial_loss(
                self.netD(generated_hr), kwargs["real_labels"]
            )

            real_features = self.criterion_content(kwargs["hr_images"])
            fake_features = self.criterion_content(generated_hr)

            content_loss_vgg = torch.abs(real_features - fake_features).mean()
            total_loss = self.content_loss * adversarial_loss + content_loss_vgg

            if self.is_l2:
                total_loss += self.l1(self.netG)

            if self.is_elastic_net:
                total_loss += self.elastic_net(self.netG)

            total_loss.backward()
            self.optimizerG.step()

        except KeyError as e:
            print("The Exception caught in (Generator) # {}".format(e).capitalize())

        except Exception as e:
            print("The Exception caught in (Generator) # {}".format(e).capitalize())

        else:
            return total_loss.item()

    def validate_model_on_test_data(self, **kwargs):
        """
        Validates the generator model using a set of test data. This method computes the loss
        for the generated high-resolution images against the actual high-resolution images using
        the defined adversarial loss function.

        Parameters:
            kwargs (dict): Contains the necessary input data for validation:
                - lr_images (torch.Tensor): A batch of low-resolution images used as input for the generator.
                - hr_images (torch.Tensor): A batch of actual high-resolution images to compare against the generated images.

        Returns:
            float: The computed loss value for the generated images when compared to the actual high-resolution images.

        Raises:
            KeyError: If any necessary keys are missing in the kwargs, such as 'lr_images' or 'hr_images'.
            Exception: Captures and logs any unexpected errors during the validation process.

        Notes:
            This method is primarily used to assess the performance of the generator on unseen data and is a crucial step
            in evaluating the effectiveness of the training process. The method assumes that the adversarial loss is appropriate
            for measuring the quality of generated images against real images.
        """
        try:
            generated_hr = self.netG(kwargs["lr_images"])

            loss = self.adversarial_loss(generated_hr, kwargs["hr_images"])

        except KeyError as e:
            print("The exception caught in # {}".format(e).capitalize())

        except Exception as e:
            print("The exception caught in # {}".format(e).capitalize())

        else:
            return loss.item()

    def save_training_images(self, **kwargs):
        """
        Saves a batch of generated high-resolution images during training for visual inspection and progress tracking.
        This method retrieves a batch of low-resolution images from the test dataloader, generates high-resolution
        images using the trained generator model, and saves them to a specified directory.

        Parameters:
            kwargs (dict): Contains additional parameters, primarily:
                - epoch (int): The current training epoch, used to label the saved image file.

        Raises:
            FileExistsError: If the specified directory for saving training images does not exist.

        Notes:
            The method assumes the existence of a global constant `TRAIN_IMAGES` which should be the path to the directory
            where training images are stored. The images are saved in a grid format with rows defined by `nrow`.
            Each image is normalized before saving to adjust the pixel values to a standard range.
        """
        lr_images, hr_images = next(iter(self.test_dataloader))
        lr_images = lr_images.to(self.device)
        hr_images = hr_images.to(self.device)

        generated_hr_images = self.netG(lr_images[0:8])

        if os.path.exists(TRAIN_IMAGES):
            save_image(
                generated_hr_images,
                os.path.join(TRAIN_IMAGES, "train_{}.png".format(kwargs["epoch"] + 1)),
                nrow=4,
                normalize=True,
            )
        else:
            raise FileExistsError("The directory should be created".capitalize())

    def show_progress(self, **kwargs):
        """
        Displays the training progress and key metrics for each epoch based on the `is_display` attribute.
        If `is_display` is True, detailed metrics are printed; otherwise, a simpler message indicating the completion
        of the epoch is shown.

        Parameters:
            kwargs (dict): Contains essential training metrics and other relevant data:
                - epoch (int): The current epoch number.
                - epochs (int): Total number of epochs planned for the training.
                - netG_loss (list): A list of loss values for the Generator during the current epoch.
                - netD_loss (list): A list of loss values for the Discriminator during the current epoch.
                - test_loss (list): A list of validation loss values for the current epoch.

        Notes:
            This method utilizes the numpy library to calculate the mean of the provided loss lists. It is essential
            that `netG_loss`, `netD_loss`, and `test_loss` are provided as lists of floats for accurate computation.
            The output format adjusts based on the `is_display` attribute, providing either detailed loss information
            or a simple epoch completion status.
        """
        if self.is_display:
            print(
                "Epochs - [{}/{}] - train_netG_loss: {:.5f} - train_netD_loss: {:.5f} - test_loss: {:.5f}".format(
                    kwargs["epoch"],
                    kwargs["epochs"],
                    np.mean(kwargs["netG_loss"]),
                    np.mean(kwargs["netD_loss"]),
                    np.mean(kwargs["test_loss"]),
                )
            )
        else:
            print(
                "Epochs - [{}/{}] is completed".format(
                    kwargs["epoch"] + 1, kwargs["epochs"]
                )
            )

    def train(self):
        """
        Executes the training process for the Generative Adversarial Network, iterating over a specified number of epochs.
        During each epoch, the method updates both the Generator and Discriminator using the training data, evaluates the model
        on test data, and logs the results. It also manages the saving of model checkpoints and training images.

        The training process involves:
        - Updating the Discriminator to better distinguish between real and generated images.
        - Updating the Generator to produce more realistic images.
        - Evaluating and recording losses for both models on test data.
        - Saving checkpoints at the end of each epoch and optionally saving generated images for visualization.

        Notes:
            This method suppresses warnings and provides detailed progress updates using the tqdm library.
            It relies on several internal methods to update models, validate on test data, and handle file operations.
        """

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

            else:
                self.save_training_images(epoch=epoch + 1)

                self.history["netG"].append(np.mean(self.netG_loss))
                self.history["netD"].append(np.mean(self.netD_loss))

            finally:
                self.show_progress(
                    epoch=epoch + 1,
                    epochs=self.epochs,
                    netG_loss=np.array(self.netG_loss).mean(),
                    netD_loss=np.array(self.netD_loss).mean(),
                    test_loss=np.array(self.test_loss).mean(),
                )

            if self.is_lr_scheduler:
                self.schedulerD.step()
                self.schedulerG.step()
        try:
            if os.path.exists(MODEL_HISTORY):
                pd.DataFrame(self.loss_track).to_csv(
                    os.path.join(MODEL_HISTORY, "history.csv")
                )
            else:
                raise FileExistsError("The directory should be created".capitalize())

        except Exception as e:
            print("The exception caught in # {}".format(e).capitalize())
        else:
            if os.path.exists(METRICS_PATH):
                dump(
                    value=self.history,
                    filename=os.path.join(METRICS_PATH, "history.pkl"),
                )
            else:
                raise FileExistsError(
                    "The directory should be created (Model History)".capitalize()
                )

    @staticmethod
    def plot_history():
        """
        Plots the training history of the Generator and Discriminator losses over epochs. This method loads the
        training history from a file and displays a line chart of the losses, providing a visual representation
        of the training process.

        Raises:
            FileExistsError: If the METRICS_PATH directory does not exist or the history file is missing,
                            indicating that the training history has not been properly saved or is otherwise
                            unavailable.

        Notes:
            This method assumes that the training history has been saved as a pickle file at a path specified by
            the global constant METRICS_PATH. It expects the history to contain keys 'netG' and 'netD' representing
            the loss values for the Generator and Discriminator, respectively, across training epochs.
        """
        if os.path.exists(METRICS_PATH):
            history = load(filename=os.path.join(METRICS_PATH, "history.pkl"))

            plt.plot(history["netG"], label="netG_loss")
            plt.plot(history["netD"], label="netD_loss")
            plt.legend()
            plt.xlabel("Epochs".capitalize())
            plt.ylabel("Loss".capitalize())
            plt.show()

        else:
            raise FileExistsError(
                "The directory should be created (Model History)".capitalize()
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trainer code for SR-GAN".title())
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of epochs".capitalize()
    )
    parser.add_argument(
        "--lr", type=float, default=0.0002, help="Learning rate".capitalize()
    )
    parser.add_argument(
        "--content_loss",
        type=float,
        default=1e-3,
        help="Content loss weight".capitalize(),
    )
    parser.add_argument(
        "--is_l1",
        type=bool,
        default=False,
        help="Use L1 loss instead of L2 loss".capitalize(),
    )
    parser.add_argument(
        "--is_l2",
        type=bool,
        default=False,
        help="Use L2 loss instead of L1 loss".capitalize(),
    )
    parser.add_argument(
        "--is_elastic_net",
        type=bool,
        default=False,
        help="Use Elastic Net loss instead of L1 and L2 loss".capitalize(),
    )
    parser.add_argument(
        "--is_lr_scheduler",
        type=bool,
        default=False,
        help="Use learning rate scheduler".capitalize(),
    )
    parser.add_argument(
        "--is_weight_init",
        type=bool,
        default=False,
        help="Use weight initialization".capitalize(),
    )
    parser.add_argument(
        "--is_weight_clip",
        type=bool,
        default=False,
        help="Use weight Clipping in netG".capitalize(),
    )
    parser.add_argument(
        "--is_display",
        type=bool,
        default=False,
        help="Display detailed loss information".capitalize(),
    )
    parser.add_argument("--device", type=str, default="mps", help="Device".capitalize())
    parser.add_argument(
        "--adam", type=bool, default=True, help="Use Adam optimizer".capitalize()
    )
    parser.add_argument(
        "--SGD", type=bool, default=False, help="Use SGD optimizer".capitalize()
    )
    parser.add_argument(
        "--beta1", type=float, default=0.5, help="Beta1 parameter".capitalize()
    )
    parser.add_argument(
        "--train", action="store_true", help="Train the model".capitalize()
    )

    args = parser.parse_args()

    if args.train:
        trainer = Trainer(
            epochs=args.epochs,
            lr=args.lr,
            content_loss=args.content_loss,
            device=args.device,
            display=args.device,
            adam=args.adam,
            SGD=args.SGD,
            beta1=args.beta1,
            is_l1=args.is_l1,
            is_l2=args.is_l2,
            is_elastic_net=args.is_elastic_net,
            is_lr_scheduler=args.is_lr_scheduler,
            is_weight_clip=args.is_weight_clip,
            is_weight_init=args.is_weight_init,
        )

        trainer.train()

        print(trainer.plot_history())

    else:
        raise Exception("Arguments should be provided".capitalize())
