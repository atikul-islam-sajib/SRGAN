import sys
import argparse
import torch
import torch.nn as nn

sys.path.append("src/")

from utils import params

from netD_helpers.input_block import InputBlock
from netD_helpers.features_block import FeatureBlock
from netD_helpers.output_block import OutputBlock


class Discriminator(nn.Module):
    """
    Defines the Discriminator model for a GAN architecture, specifically designed for tasks like super-resolution
    (SRGAN). The discriminator aims to differentiate between real high-resolution images and fake images produced
    by the generator.

    The network consists of an initial input block, a series of feature blocks for extracting and downscaling features,
    followed by an adaptive max pooling layer to reduce the spatial dimensions to 1x1, and a final output block that
    classifies the input as real or fake.

    Attributes:
        in_channels (int): The number of channels in the input images.
        out_channels (int): The initial number of output channels, which is scaled in subsequent layers.
        filters (int): A copy of `out_channels` used for output block initialization.
        layers (list): A list of feature blocks dynamically added based on `num_repetitive`.
        input (InputBlock): The initial block to process the input image.
        features (nn.Sequential): Sequential container of feature blocks.
        avg_pool (nn.AdaptiveMaxPool2d): Adaptive max pooling layer to reduce spatial dimensions.
        output (OutputBlock): Final block to output the discriminator's classification.
    """

    def __init__(self, in_channels=3, out_channels=64):
        """
        Initializes the Discriminator model with specified configurations for its input and feature extraction layers.

        Parameters:
            in_channels (int, optional): The number of input channels. Defaults to None.
            out_channels (int, optional): The initial number of output channels for the first layer. Defaults to None.
        """
        super(Discriminator, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filters = out_channels

        try:
            self.kernel_size = params()["netD"]["kernel_size"]
            self.stride = params()["netD"]["stride"]
            self.padding = params()["netD"]["padding"]
            self.num_repetitive = params()["netD"]["num_repetitive"]

        except Exception as e:
            print("The exception caught in the section # {}".format(e).capitalize())

        else:
            self.layers = []

        self.input = InputBlock(
            in_channels=self.in_channels, out_channels=self.out_channels
        )

        for index in range(self.num_repetitive):
            if index % 2:
                self.layers.append(
                    FeatureBlock(
                        in_channels=self.out_channels,
                        out_channels=self.out_channels * 2,
                    )
                )

                self.out_channels = self.out_channels * 2

            else:
                self.layers.append(
                    FeatureBlock(
                        in_channels=self.out_channels, out_channels=self.out_channels
                    )
                )

                self.out_channels = self.out_channels

        self.features = nn.Sequential(*self.layers)

        self.avg_pool = nn.AdaptiveMaxPool2d(output_size=1)

        self.output = OutputBlock(in_channels=self.filters, out_channels=1)

    def forward(self, x):
        """
        Defines the forward pass through the Discriminator model.

        Parameters:
            x (torch.Tensor): The input tensor representing the images to be classified.

        Returns:
            torch.Tensor: A flattened tensor with the discriminator's classification scores.
        """
        input = self.input(x)
        features = self.features(input)
        output = self.output(self.avg_pool(features))

        return output.view(-1)

    @staticmethod
    def total_params(model):
        """
        Calculates the total number of trainable parameters in the given model.

        Parameters:
            model (torch.nn.Module): The PyTorch model for which the total parameters need to be calculated.

        Returns:
            int: Total number of parameters in the model.

        Raises:
            Exception: If the model is not provided or is None.
        """
        if model:
            return sum(params.numel() for params in model.parameters())
        else:
            raise Exception("Model should be provided".capitalize())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Discriminator for SRGAN".title())
    parser.add_argument("--in_channels", type=int, default=1, help="Input channels")
    parser.add_argument("--out_channels", type=int, default=64, help="Output channels")

    parser.add_argument(
        "--netD", action="store_true", help="Generate a generator".capitalize()
    )

    args = parser.parse_args()

    if args.netD:
        netD = Discriminator(
            in_channels=args.in_channels, out_channels=args.out_channels
        )

        print(netD)

        images = torch.randn(4, 3, 64, 64)

        print(netD(images).shape)

        print(
            "Total params of the Discriminator model is # {}".format(
                Discriminator.total_params(model=netD)
            )
        )

    else:
        raise Exception("Arguments should be passed".capitalize())
