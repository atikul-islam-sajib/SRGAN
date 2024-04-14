import sys
import os
import argparse
import yaml
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torchvision import transforms

sys.path.append("src/")

from config import SINGLE_IMAGE, BATCH_IMAGE, PROCESSED_DATA_PATH
from utils import load
from generator import Generator
from charts import Test


class Inference(Test):
    """
    Extends the Test class to provide inference capabilities for a trained SR-GAN model. This class handles
    loading and running inference on single images or batches of images using the super-resolution generator
    network to produce high-resolution outputs.

    Attributes:
        image (str): Path to a single image file for SRGAN inference.
        model (torch.nn.Module): Pre-trained model for inference.
        device (str): The device on which the model will perform the inference.

    Methods:
        model_init: Initializes and loads the generator model for inference.
        image_transformation: Returns a composed set of transformations for input images.
        image_preprocessing: Applies transformations to the input image.
        srgan_single: Performs super-resolution on a single image and saves the output.
        srgan_batch: Processes a batch of images for super-resolution and saves the outputs.
    """

    def __init__(self, image=None, model=None, device="mps"):
        """
        Initializes the Inference class with the specified image, model, and computation device.

        Parameters:
            image (str): Path to the image for single image inference.
            model (str): Path to the pre-trained model file.
            device (str): Device to use for inference ('cuda', 'mps', 'cpu').
        """
        super(Inference, self).__init__(model=model, device=device)
        self.image = image
        self.netG = self.model_init()

        try:
            with open("./trained_params.yml", "r") as file:
                self.config = yaml.safe_load(file)

        except Exception as e:
            print("The exception is: ", e)

    def model_init(self):
        """
        Initializes or loads the generator model from a provided model path or selects the best model.

        Returns:
            torch.nn.Module: The initialized or loaded generator model.
        """
        self.netG = Generator().to(self.device)

        if self.model:
            self.netG.load_state_dict(torch.load(self.model)["netG"])
        else:
            self.netG.load_state_dict(self.select_best_model())

        return self.netG

    def image_transformation(self):
        """
        Creates a series of transformations for preprocessing the images for the generator network.

        Returns:
            torchvision.transforms.Compose: The transformation pipeline.
        """
        return transforms.Compose(
            [
                transforms.Resize(
                    (
                        self.config["params"]["dataloader"]["image_size"],
                        self.config["params"]["dataloader"]["image_size"],
                    )
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def image_preprocessing(self, **kwargs):
        """
        Applies predefined transformations to the image.

        Parameters:
            kwargs (dict): Contains the image to be processed.

        Returns:
            torch.Tensor: The transformed image tensor ready for the generator.
        """
        return self.image_transformation()(kwargs["image"]).to(self.device)

    def srgan_single(self):
        """
        Processes a single image, applies the generator model, and saves the output as a high-resolution image.

        Parameters:
            kwargs (dict): Contains the image to be processed.

        Returns:
            torch.Tensor: The transformed image tensor ready for the generator.
        """
        self.image = Image.fromarray(cv2.imread(self.image))
        self.image = self.image_preprocessing(image=self.image)

        hr_image = self.netG(self.image.unsqueeze(0))
        hr_image = hr_image.squeeze(0)
        hr_image = hr_image.permute(1, 2, 0).squeeze().cpu().detach().numpy()
        hr_image = self.image_normalized(image=hr_image)

        plt.imshow(hr_image, cmap="gray")

        if os.path.exists(SINGLE_IMAGE):
            plt.savefig(os.path.join(SINGLE_IMAGE, "result.png"))
        else:
            raise Exception("The directory does not exist".capitalize())

    def srgan_batch(self):
        """
        Processes multiple images in a batch from a predefined dataset, applies super-resolution
        using the generator model, and saves the high-resolution images to a specified directory.
        This method is designed to handle batches of images for efficient processing.

        Raises:
            Exception: If the dataset or the directory for saving batch images is not found.

        Notes:
            - This method assumes the existence of a 'test_dataloader.pkl' file within the directory specified by
              PROCESSED_DATA_PATH, which contains pre-processed low-resolution images for inference.
            - Output high-resolution images are saved sequentially in the directory specified by BATCH_IMAGE.
            - Images are normalized post-generation to enhance visual quality before saving.
        """
        if os.path.exists(PROCESSED_DATA_PATH):
            self.dataloader = load(
                filename=os.path.join(PROCESSED_DATA_PATH, "test_dataloader.pkl")
            )

            if os.path.exists(BATCH_IMAGE):
                for lr_images, _ in self.dataloader:
                    lr_images = lr_images.to(self.device)
                    for index, image in enumerate(lr_images):
                        gen_hr_images = self.netG(image.unsqueeze(0))
                        gen_hr_images = gen_hr_images.squeeze(0)
                        gen_hr_images = (
                            gen_hr_images.permute(1, 2, 0)
                            .squeeze()
                            .cpu()
                            .detach()
                            .numpy()
                        )
                        gen_hr_images = self.image_normalized(image=gen_hr_images)

                        plt.imshow(gen_hr_images, cmap="gray")
                        plt.axis("off")

                        plt.savefig(
                            os.path.join(BATCH_IMAGE, "result{}.png".format(index))
                        )
        else:
            raise Exception("Dataset not found".capitalize())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference for SR-GAN".title())

    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to the image for SRGAN".capitalize(),
    )
    parser.add_argument(
        "--model", type=str, default=None, help="Path to the best model".capitalize()
    )
    parser.add_argument("--device", type=str, default="mps", help="Device".capitalize())
    parser.add_argument(
        "--single", action="store_true", help="Single image inference".capitalize()
    )
    parser.add_argument(
        "--batch", action="store_true", help="Batch image inference".capitalize()
    )
    args = parser.parse_args()

    if args.single:
        inference = Inference(image=args.image, model=args.model, device=args.device)
        inference.srgan_single()

    elif args.batch:
        inference = Inference(image=args.image, model=args.model, device=args.device)
        inference.srgan_batch()
