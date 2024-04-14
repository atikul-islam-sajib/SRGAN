import sys
import os
import argparse
import imageio
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

sys.path.append("src/")

from config import PROCESSED_DATA_PATH, BEST_MODELS, GIF_FILE, TRAIN_IMAGES, TEST_IMAGES
from utils import device_init, load
from generator import Generator


class Test:
    """
    A class designed to handle testing and evaluation of a neural network model on a specified device.

    This class is responsible for initializing the model, setting the computation device, and managing the
    best-performing model across tests.

    Attributes:
        device (str): The device on which the model will be tested and evaluated.
        model (torch.nn.Module, optional): The model to be tested. If provided, it is used directly; otherwise,
                                           a new Generator model is initialized.
        infinity (float): A placeholder for tracking the lowest loss or best performance metrics.
        best_model (torch.nn.Module): To store the best model based on performance metrics.
        netG (torch.nn.Module): The generator model, initialized and moved to the specified device.
    """

    def __init__(self, device="mps", model=None):
        """
        Initializes the Test class with the specified computation device and model.

        Parameters:
            device (str): The computation device ('cuda', 'mps', 'cpu') to which the model should be transferred.
                          Default is 'mps' for Apple's Metal Performance Shaders on supported hardware.
            model (torch.nn.Module, optional): A pre-initialized model to be used for testing. If None, a new
                                               Generator model is created and used.

        Notes:
            The `device_init` function is assumed to handle the specifics of device initialization and setup,
            including error handling for unsupported devices. The class is designed to work with models that
            have a Generator-like architecture, but it can be adapted for other model types by passing a
            different model to the constructor.
        """
        self.device = device_init(device=device)
        self.model = model
        self.infinity = float("inf")

        self.best_model = None
        self.netG = Generator().to(self.device)

    def load_dataset(self):
        """
        Loads the test dataset from a predefined path. This method retrieves the test dataloader
        from a pickle file stored in the processed data directory.

        Returns:
            DataLoader: The test dataloader loaded from the specified pickle file.

        Raises:
            Exception: If the dataset file does not exist in the processed data path, indicating
                        that the dataset has not been prepared or is otherwise unavailable.

        Notes:
            The method expects that the test dataset has been previously processed and saved as a pickle file
            at a path defined by the global constant PROCESSED_DATA_PATH. It assumes that the dataloader contains
            all necessary preprocessing and is ready for use in model evaluation or testing.
        """
        if os.path.exists(PROCESSED_DATA_PATH):
            return load(
                filename=os.path.join(PROCESSED_DATA_PATH, "test_dataloader.pkl")
            )
        else:
            raise Exception("Dataset not found".capitalize())

    def select_best_model(self):
        """
        Selects the best-performing model based on a loss metric from a directory of saved models. This method iterates
        over saved model files, loads each model's loss, and updates the best model if it finds a lower loss than previously
        recorded. The method assumes models are saved with their loss stored in a dictionary under 'netG_loss'.

        Returns:
            torch.nn.Module: The best model found based on the lowest loss.

        Raises:
            Exception: If the directory supposed to contain the best models does not exist, indicating a misconfiguration
                        or that no models have been saved as 'best' models yet.

        Notes:
            This method is designed to be run after model training has concluded and a series of models have been evaluated
            and saved as 'best' under certain criteria. The directory containing these models is expected to be specified
            by the global constant BEST_MODELS. Non-model files (like '.DS_Store') are ignored.
        """
        if os.path.exists(BEST_MODELS):
            for model in os.listdir(BEST_MODELS):
                if model != ".DS_Store":
                    model_path = os.path.join(BEST_MODELS, model)
                    model_loss = torch.load(model_path)
                    model_loss = model_loss["netG_loss"]

                    if self.infinity > model_loss:
                        self.infinity = model_loss

                        self.best_model = torch.load(model_path)
                        self.best_model = self.best_model["netG"]

                return self.best_model
        else:
            raise Exception("Best models not found".capitalize())

    def create_gif(self):
        """
        Creates an animated GIF from a series of images stored in a specified directory. This method loads each image
        from the directory, compiles them into a GIF, and saves the GIF to a predefined file location.

        Raises:
            Exception: If the directory for storing training images or the target directory for the GIF does not exist,
                        indicating that the necessary preliminary files or directories are missing.

        Notes:
            This method assumes that there is a global constant `TRAIN_IMAGES` which specifies the directory containing
            the images to be used in the GIF, and `GIF_FILE` which specifies the directory where the GIF should be saved.
            It is expected that the images are compatible for GIF creation and are stored in a format readable by `imageio`.
        """
        if (os.path.exists(GIF_FILE)) and (os.path.exists(TRAIN_IMAGES)):
            self.images = []

            for image in os.listdir(TRAIN_IMAGES):
                self.images.append(imageio.imread(os.path.join(TRAIN_IMAGES, image)))

            imageio.mimsave(os.path.join(GIF_FILE, "train_gif.gif"), self.images)

        else:
            raise Exception("GIF file not found".capitalize())

    def image_normalized(self, **kwargs):
        """
        Normalizes an image tensor to have pixel values ranging from 0 to 1. This method adjusts the image data
        by applying min-max normalization, which scales the pixel values based on the minimum and maximum values
        found in the image tensor.

        Parameters:
            kwargs (dict): Contains the image tensor to be normalized:
                - image (torch.Tensor): The image tensor that needs normalization.

        Returns:
            torch.Tensor: A new tensor with the same dimensions as the input, where each pixel value is scaled
                          to a range between 0 and 1.

        Raises:
            KeyError: If the 'image' key is missing from the kwargs, indicating that no image tensor was provided.

        Notes:
            Normalization is a common preprocessing step for image data before it is inputted into neural network models.
            This method expects that the 'image' tensor is provided and that it contains numeric data suitable for
            min-max scaling.
        """
        return (kwargs["image"] - kwargs["image"].min()) / (
            kwargs["image"].max() - kwargs["image"].min()
        )

    def plot_images(self, **kwargs):
        """
        Plots and saves a grid of low-resolution, high-resolution, and generated high-resolution images. This method
        fetches a batch of images from the test dataset, generates corresponding high-resolution images using the
        trained model, and displays them side by side for comparison.

        The method will use the provided model to generate images if available, or it will attempt to load the best model
        automatically. It normalizes images before displaying to enhance visual quality.

        Parameters:
            **kwargs: Accepts optional parameters but primarily used for flexible function extension in the future.

        Raises:
            Exception: If the directory to save the images does not exist and cannot be created.

        Notes:
            This method assumes that the `TEST_IMAGES` directory is used to store output images and checks for its
            existence before saving. If it doesn't exist, the method attempts to create the directory and raises an
            exception if it fails. This is a comprehensive method for visual evaluation of the model's performance
            on generating high-resolution images from low-resolution inputs.
        """

        plt.figure(figsize=(80, 40))

        lr_images, hr_images = next(iter(self.load_dataset()))
        lr_images = lr_images.to(self.device)
        hr_images = hr_images.to(self.device)

        if self.model:
            self.netG.load_state_dict(torch.load(self.model)["netG"])
        else:
            self.netG.load_state_dict(self.select_best_model())

        generated_hr = self.netG(lr_images)

        for index, image in enumerate(generated_hr):
            lr = lr_images[index].permute(1, 2, 0).squeeze().cpu().detach().numpy()
            hr = hr_images[index].permute(1, 2, 0).squeeze().cpu().detach().numpy()
            gen_image = image.permute(1, 2, 0).squeeze().cpu().detach().numpy()

            lr = self.image_normalized(image=lr)
            hr = self.image_normalized(image=hr)
            gen_image = self.image_normalized(image=gen_image)

            plt.subplot(3 * 8, 3 * 8, 3 * index + 1)
            plt.imshow(lr, cmap="gray")
            plt.title("lr".lower())
            plt.axis("off")

            plt.subplot(3 * 8, 3 * 8, 3 * index + 2)
            plt.imshow(hr, cmap="gray")
            plt.title("hr".lower())
            plt.axis("off")

            plt.subplot(3 * 8, 3 * 8, 3 * index + 3)
            plt.imshow(gen_image, cmap="gray")
            plt.title("sr".upper())
            plt.axis("off")

        plt.tight_layout()

        if os.path.exists(TEST_IMAGES):
            plt.savefig(os.path.join(TEST_IMAGES, "result.png"))

        else:
            os.mkdir(TEST_IMAGES)
            raise Exception(
                "Could not create the image due to folder is not found".capitalize()
            )

        plt.show()

    def plot(self):
        """
        Orchestrates the visualization of generated images and the creation of a GIF from those images. This method
        first attempts to plot images using a model and then, if successful, proceeds to create an animated GIF from the
        images stored during the plotting process.

        The method relies on the internal methods `plot_images` and `create_gif` to perform its operations.

        Raises:
            Exception: Captures and logs any exceptions that occur during the plotting or GIF creation processes, providing
                        details about the nature of the failure.

        Notes:
            This method is a higher-level orchestration method that combines the functionalities of image plotting and
            GIF creation to provide a comprehensive visual output of the model's performance. It is designed to be used
            after the model has generated a series of images which are then used to create a visual narrative of the
            training or testing process.
        """
        try:
            self.plot_images(netG=self.netG)

        except Exception as e:
            print("The exception is: ", e)

        else:
            self.create_gif()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the model".title())
    parser.add_argument("--device", type=str, default="mps", help="Device".capitalize())
    parser.add_argument(
        "--model", type=str, default=None, help="Path to the best model".capitalize()
    )
    parser.add_argument(
        "--test", action="store_true", help="Test the model".capitalize()
    )

    args = parser.parse_args()

    if args.test:
        test = Test(device=args.device, model=args.model)
        test.plot()
    else:
        raise Exception("Arguments should be provided precisely".capitalize())
