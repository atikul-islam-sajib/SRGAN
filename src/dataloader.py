import sys
import argparse
import os
import cv2
import zipfile
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

sys.path.append("src/")

from utils import dump, load, params
from config import RAW_DATA_PATH, PROCESSED_DATA_PATH, FILES_PATH


class Loader(Dataset):
    """
    Loader class for processing and creating data loaders for SRGAN training and testing datasets.
    This class handles image loading, preprocessing, and transformations to create PyTorch DataLoaders.

    Attributes:
        image_path (str): Path to the zip file containing raw images.
        in_channels (int): Number of channels in the input images.
        batch_size (int): Batch size for the DataLoader.
        image_size (int): Size to which input images are resized.
    """

    def __init__(
        self,
        image_path=None,
        in_channels=3,
        batch_size=1,
        image_size=64,
        is_sub_samples=False,
    ):
        """
        Initializes the Loader with specified configurations for image preprocessing and DataLoader creation.

        Parameters:
            image_path (str, optional): Path to the zip file containing raw images. Defaults to None.
            in_channels (int, optional): Number of channels in the input images. Defaults to 3.
            batch_size (int, optional): Batch size for the DataLoader. Defaults to 1.
            image_size (int, optional): Size to which input images are resized. Defaults to 64.
        """
        self.image_path = image_path
        self.batch_size = batch_size
        self.in_channels = in_channels
        self.image_size = image_size
        self.is_sub_samples = is_sub_samples

        self.train_images = list()
        self.train_labels = list()
        self.test_images = list()
        self.test_labels = list()

        self.norm_value = params()["dataloader"]["images"]["normalized"]

    def image_transforms(self, lr_images=True):
        """
        Constructs a composition of image transformations including resizing, tensor conversion, and normalization.

        Parameters:
            lr_images (bool): If True, images are considered as low-resolution and resized to `image_size`.
                              If False, images are considered high-resolution and resized to `image_size * 4`.
                              Defaults to True.

        Returns:
            torchvision.transforms.Compose: The composite transformations for image preprocessing.
        """
        return transforms.Compose(
            [
                (
                    transforms.Resize((self.image_size, self.image_size))
                    if lr_images
                    else transforms.Resize((self.image_size * 4, self.image_size * 4))
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[self.norm_value, self.norm_value, self.norm_value],
                    std=[self.norm_value, self.norm_value, self.norm_value],
                ),
            ]
        )

    def unzip_folder(self):
        """
        Unzips the folder containing raw images into the specified directory for raw data.
        """
        if os.path.exists(RAW_DATA_PATH):
            with zipfile.ZipFile(self.image_path, "r") as zip_file:
                zip_file.extractall(os.path.join(RAW_DATA_PATH, "images"))
        else:
            raise Exception("Raw data folder not found".capitalize())

    def process_images(self, **kwargs):
        """
        Applies image transformations to a single image.

        Parameters:
            kwargs (dict): Contains parameters for image processing including 'lr_images' to specify
                           if the image is low-resolution and 'images' for the image data.

        Returns:
            torch.Tensor: The transformed image tensor.
        """
        return self.image_transforms(kwargs["lr_images"])(
            Image.fromarray(kwargs["images"])
        )

    def get_subsample_of_dataloader(self):
        """
        Reduces the size of the data loaders by subsampling the datasets. The training data is reduced to 20% of its
        original size, and the testing data is reduced to 50% of its original size.

        This method modifies the internal state of the class by updating the train and test datasets to their new, smaller sizes.

        Returns:
            dict: A dictionary containing the subsampled datasets:
                  - 'train_images': Subsampled training images.
                  - 'train_labels': Subsampled training labels.
                  - 'test_images': Subsampled testing images.
                  - 'test_labels': Subsampled testing labels.
        """

        train_samples = len(self.train_images) // 5
        test_samples = len(self.test_images) // 2

        self.train_images = self.train_images[:train_samples]
        self.train_labels = self.train_labels[:train_samples]

        self.test_images = self.test_images[:test_samples]
        self.test_labels = self.test_labels[:test_samples]

        return {
            "train_images": self.train_images,
            "train_labels": self.train_labels,
            "test_images": self.test_images,
            "test_labels": self.test_labels,
        }

    def extract_images(self):
        """
        Extracts and processes images from the specified directory, splitting them into training and testing datasets.

        Returns:
            dict: A dictionary containing processed training and testing images and labels.
        """
        if os.path.exists(RAW_DATA_PATH):
            image_path = os.path.join(RAW_DATA_PATH, "images")

            train_images = os.path.join(image_path, os.listdir(image_path)[0])
            test_images = os.path.join(image_path, os.listdir(image_path)[1])

            for index, path in enumerate([train_images, test_images]):
                categories = os.listdir(path)

                for category in categories:
                    images = os.path.join(path, category)

                    for image in os.listdir(images):
                        self.image_path = os.path.join(images, image)
                        self.image = cv2.imread(self.image_path)

                        if self.image is not None:
                            if index == 0:
                                self.train_images.append(
                                    self.process_images(
                                        lr_images=True, images=self.image
                                    )
                                )

                                self.train_labels.append(
                                    self.process_images(
                                        lr_images=False, images=self.image
                                    )
                                )
                            else:
                                self.test_images.append(
                                    self.process_images(
                                        lr_images=True, images=self.image
                                    )
                                )

                                self.test_labels.append(
                                    self.process_images(
                                        lr_images=False, images=self.image
                                    )
                                )
                        else:
                            continue

            if self.is_sub_samples == True:
                self.images = self.get_subsample_of_dataloader()

                return {
                    "train_images": self.images["train_images"],
                    "train_labels": self.images["train_labels"],
                    "test_images": self.images["test_images"],
                    "test_labels": self.images["test_labels"],
                }

            else:
                return {
                    "train_images": self.train_images,
                    "train_labels": self.train_labels,
                    "test_images": self.test_images,
                    "test_labels": self.test_labels,
                }
        else:
            raise Exception("Raw data folder not found".capitalize())

    def create_dataloader(self):
        """
        Creates and dumps training and testing DataLoaders for the SRGAN project, handling images extraction
        and processing.
        """
        try:
            images = self.extract_images()

        except Exception as e:
            print("Error in extracting images".capitalize())

        else:
            train_dataloader = DataLoader(
                dataset=list(zip(images["train_images"], images["train_labels"])),
                batch_size=self.batch_size,
                shuffle=True,
            )
            test_dataloader = DataLoader(
                dataset=list(zip(images["test_images"], images["test_labels"])),
                batch_size=self.batch_size * 64,
                shuffle=True,
            )

            if os.path.join(PROCESSED_DATA_PATH):
                try:
                    dump(
                        value=train_dataloader,
                        filename=os.path.join(
                            PROCESSED_DATA_PATH, "train_dataloader.pkl"
                        ),
                    )
                except Exception as e:
                    print("Error in dumping train dataloader".capitalize())

                try:
                    dump(
                        value=test_dataloader,
                        filename=os.path.join(
                            PROCESSED_DATA_PATH, "test_dataloader.pkl"
                        ),
                    )
                except Exception as e:
                    print("Error in dumping test dataloader".capitalize())
            else:
                raise Exception("Processed data folder not found".capitalize())

    @staticmethod
    def image_normalization(**kwargs):
        """
        Normalizes an image tensor to the range [0, 1].

        Parameters:
            kwargs (dict): Contains the 'image' tensor for normalization.

        Returns:
            np.ndarray: The normalized image array.
        """
        image = kwargs["image"].squeeze().permute(1, 2, 0)
        image = image.cpu().detach().numpy()
        image = (image - image.min()) / (image.max() - image.min())

        return image

    @staticmethod
    def display_images():
        """
        Displays a batch of low-resolution and corresponding high-resolution images using matplotlib.
        """
        if os.path.exists(PROCESSED_DATA_PATH):

            lr_images, hr_images = next(
                iter(
                    load(
                        filename=os.path.join(
                            PROCESSED_DATA_PATH, "test_dataloader.pkl"
                        )
                    )
                )
            )

            plt.figure(figsize=(40, 25))

            for index, (lr_image, hr_image) in enumerate(zip(lr_images, hr_images)):

                image = Loader.image_normalization(image=lr_image)

                plt.subplot(2 * 8, 2 * 8, 2 * index + 1)
                plt.imshow(image.squeeze(), cmap="gray")
                plt.title("lr_image".lower())
                plt.axis("off")

                image = Loader.image_normalization(image=hr_image)

                plt.subplot(2 * 8, 2 * 8, 2 * index + 2)
                plt.imshow(image.squeeze(), cmap="gray")
                plt.title("hr_image".lower())
                plt.axis("off")

            plt.tight_layout()
            plt.show()

    @staticmethod
    def details_dataset():
        """
        Generates and returns details of the dataset including quantity and size of training/testing images.

        Returns:
            str: Details of the dataset in string format.
        """
        if os.path.exists(PROCESSED_DATA_PATH):
            train_dataloader = load(
                filename=os.path.join(PROCESSED_DATA_PATH, "train_dataloader.pkl")
            )
            test_dataloader = load(
                filename=os.path.join(PROCESSED_DATA_PATH, "test_dataloader.pkl")
            )

            train = sum(data.size(0) for data, _ in train_dataloader)
            test = sum(data.size(0) for data, _ in test_dataloader)

            train_lr_images, train_hr_images = next(iter(train_dataloader))
            test_lr_images, test_hr_images = next(iter(test_dataloader))

            details = pd.DataFrame(
                {
                    "Train Images": [train],
                    "Test Images": [test],
                    "Train Images Size(lr)": [train_lr_images.squeeze().size()],
                    "Test Images Size(lr)": [test_lr_images.squeeze().size()],
                    "Train Images Size(hr)": [train_hr_images.squeeze().size()],
                    "Test Images Size(hr)": [test_hr_images.squeeze().size()],
                },
                index=["Quantity"],
            ).T.to_string()

            try:
                if os.path.exists(FILES_PATH):
                    details.to_csv(os.path.join(FILES_PATH, "details.csv"), index=False)

            except Exception as e:
                print("Error in dumping train dataloader".capitalize())

            return details

        else:
            raise Exception("Processed data folder not found".capitalize())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Loader for SRGAN".title())
    parser.add_argument(
        "--image_path", type=str, help="Path to the image for SRGAN".capitalize()
    )
    parser.add_argument(
        "--batch_size", type=int, help="Batch size for the dataloader".capitalize()
    )
    parser.add_argument(
        "--image_size", type=int, help="Image size for the dataloader".capitalize()
    )
    parser.add_argument(
        "--is_sub_samples",
        type=bool,
        default=False,
        help="Image size for the dataloader".capitalize(),
    )

    args = parser.parse_args()

    if args.image_path and args.batch_size and args.image_size:
        loader = Loader(
            image_path=args.image_path,
            batch_size=args.batch_size,
            image_size=args.image_size,
            is_sub_samples=args.is_sub_samples,
        )
        try:
            loader.unzip_folder()
            loader.create_dataloader()

        except Exception as e:
            print("Error in creating dataloader".capitalize())

        finally:
            print(loader.details_dataset())
            loader.display_images()

    else:
        raise Exception("Missing arguments".capitalize())
