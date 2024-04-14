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
    def __init__(self, device="mps", model=None):
        self.device = device_init(device=device)
        self.model = model
        self.infinity = float("inf")

        self.best_model = None
        self.netG = Generator().to(self.device)

    def load_dataset(self):
        if os.path.exists(PROCESSED_DATA_PATH):
            return load(
                filename=os.path.join(PROCESSED_DATA_PATH, "test_dataloader.pkl")
            )
        else:
            raise Exception("Dataset not found".capitalize())

    def select_best_model(self):
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
        if (os.path.exists(GIF_FILE)) and (os.path.exists(TRAIN_IMAGES)):
            self.images = []

            for image in os.listdir(TRAIN_IMAGES):
                self.images.append(imageio.imread(os.path.join(TRAIN_IMAGES, image)))

            imageio.mimsave(os.path.join(GIF_FILE, "train_gif.gif"), self.images)

        else:
            raise Exception("GIF file not found".capitalize())

    def image_normalized(self, **kwargs):
        return (kwargs["image"] - kwargs["image"].min()) / (
            kwargs["image"].max() - kwargs["image"].min()
        )

    def plot_images(self, **kwargs):

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
