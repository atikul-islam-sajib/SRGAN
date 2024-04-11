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

from utils import dump, load
from config import RAW_DATA_PATH, PROCESSED_DATA_PATH


class Loader(Dataset):
    def __init__(self, image_path=None, in_channels=3, batch_size=1, image_size=64):
        self.image_path = image_path
        self.batch_size = batch_size
        self.in_channels = in_channels
        self.image_size = image_size

        self.train_images = list()
        self.train_labels = list()
        self.test_images = list()
        self.test_labels = list()

    def image_normalized(self, lr_images=True):
        return transforms.Compose(
            [
                (
                    transforms.Resize((self.image_size, self.image_size))
                    if lr_images
                    else transforms.Resize((self.image_size * 4, self.image_size * 4))
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def unzip_folder(self):
        if os.path.exists(RAW_DATA_PATH):
            with zipfile.ZipFile(self.image_path, "r") as zip_file:
                zip_file.extractall(os.path.join(RAW_DATA_PATH, "images"))
        else:
            raise Exception("Raw data folder not found".capitalize())

    def process_images(self, **kwargs):
        return self.image_normalized(kwargs["lr_images"])(
            Image.fromarray(kwargs["images"])
        )

    def extract_images(self):
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
            return {
                "train_images": self.train_images,
                "train_labels": self.train_labels,
                "test_images": self.test_images,
                "test_labels": self.test_labels,
            }
        else:
            raise Exception("Raw data folder not found".capitalize())

    def create_dataloader(self):
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
                batch_size=self.batch_size,
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
    def display_images():
        if os.path.exists(PROCESSED_DATA_PATH):
            images = list()
            labels = list()

            train_images = load(
                filename=os.path.join(PROCESSED_DATA_PATH, "train_dataloader.pkl")
            )

            for index, (image, label) in enumerate(train_images):

                if index != 64:
                    images.append(image)
                    labels.append(label)
                else:
                    break

            plt.figure(figsize=(40, 25))

            for index, (lr_image, hr_image) in enumerate(zip(images, labels)):

                lr_image = lr_image.squeeze().permute(1, 2, 0)
                lr_image = lr_image.cpu().detach().numpy()
                lr_image = (lr_image - lr_image.min()) / (
                    lr_image.max() - lr_image.min()
                )

                plt.subplot(2 * 8, 2 * 8, 2 * index + 1)
                plt.imshow(lr_image, cmap="gray")
                plt.title("lr_image".lower())
                plt.axis("off")

                hr_image = hr_image.squeeze().permute(1, 2, 0)
                hr_image = hr_image.cpu().detach().numpy()
                hr_image = (hr_image - hr_image.min()) / (
                    hr_image.max() - hr_image.min()
                )

                plt.subplot(2 * 8, 2 * 8, 2 * index + 2)
                plt.imshow(hr_image, cmap="gray")
                plt.title("hr_image".lower())
                plt.axis("off")

            plt.tight_layout()
            plt.show()

    @staticmethod
    def details_dataset():
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
                if os.path.exists("FILES_PATH"):
                    details.to_csv(
                        os.path.join("FILES_PATH", "details.csv"), index=False
                    )
            except Exception as e:
                print("Error in dumping train dataloader".capitalize())

            return details

        else:
            raise Exception("Processed data folder not found".capitalize())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Loader for SRGAN".title())
    parser.add_argument("--image_path", type=str, help="Path to the image".capitalize())
    parser.add_argument(
        "--batch_size", type=int, help="Batch size for the dataloader".capitalize()
    )
    parser.add_argument(
        "--image_size", type=int, help="Image size for the dataloader".capitalize()
    )

    args = parser.parse_args()

    if args.image_path and args.batch_size and args.image_size:
        loader = Loader(
            image_path=args.image_path,
            batch_size=args.batch_size,
            image_size=args.image_size,
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
