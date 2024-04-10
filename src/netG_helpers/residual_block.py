import sys
import argparse
from collections import OrderedDict
import torch
import torch.nn as nn

sys.path.append("src/")

from utils import params


class ResidualBlock(nn.Module):
    """
    Implements a residual block for a neural network, specifically designed for 'netG'. A residual block helps in
    preventing the vanishing gradient problem in deep networks by allowing an alternate shortcut path for the gradient.

    This block consists of two convolutional layers each followed by batch normalization and a PReLU activation layer,
    with a skip connection that adds the input directly to the block's output.

    Attributes:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        index (int): An identifier for the layers within the block, used for layer naming.
        kernel_size (int): Kernel size for the convolutional layers.
        stride (int): Stride for the convolutional layers.
        padding (int): Padding for the convolutional layers.
        model (nn.Sequential): The sequential container that comprises the block's layers.

    Examples:
        >>> residual_block = ResidualBlock(in_channels=64, out_channels=64, index=0)
        >>> images = torch.randn(1, 64, 64, 64)
        >>> output = residual_block(images)
        >>> print(output.size())
        torch.Size([1, 64, 64, 64])
    """
    def __init__(self, in_channels=None, out_channels=None, index=None):
        """
        Initializes the ResidualBlock with specified configurations.
        
        Parameters:
            in_channels (int, optional): The number of channels in the input tensor. Defaults to None.
            out_channels (int, optional): The number of channels in the output tensor. Defaults to None.
            index (int, optional): Index for naming the layers uniquely within the block. Defaults to None.
        """
        super(ResidualBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.index = index

        self.kernel_size = params()["netG"]["block"]["kernel_size"]
        self.stride = params()["netG"]["block"]["stride"]
        self.padding = params()["netG"]["block"]["padding"]

        try:
            self.model = self.residual_block()
        except Exception as e:
            print("Residual block not implemented")

    def residual_block(self):
        """
        Constructs the layers for the residual block, including convolutional, batch normalization, and PReLU layers.

        Returns:
            nn.Sequential: The sequential model containing the block's layers.
        """
        layers = OrderedDict()

        layers["conv{}".format(self.index)] = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
        )
        layers["batchnorm{}".format(self.index)] = nn.BatchNorm2d(
            num_features=self.out_channels
        )

        layers["PReLU{}".format(self.index)] = nn.PReLU()
        layers["conv{}".format(self.index + 1)] = nn.Conv2d(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
        )
        layers["batchnorm{}".format(self.index + 1)] = nn.BatchNorm2d(
            num_features=self.out_channels
        )

        return nn.Sequential(layers)

    def forward(self, x=None):
        """
        Defines the forward pass of the ResidualBlock with a skip connection that adds the input to the output.

        Parameters:
            x (torch.Tensor, optional): The input tensor to the block. Defaults to None.

        Raises:
            Exception: If the input tensor x is None, indicating that the block's implementation is incomplete.

        Returns:
            torch.Tensor: The output tensor after adding the input to the processed output from the block.
        """
        if x is not None:
            return x + self.model(x)
        else:
            raise Exception("Residual block not implemented")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Residual block for netG".capitalize())
    parser.add_argument(
        "--in_channels",
        type=int,
        default=64,
        help="Define the in channels".capitalize(),
    )
    parser.add_argument(
        "--out_channels",
        type=int,
        default=64,
        help="Define the out channels".capitalize(),
    )

    args = parser.parse_args()

    if args.in_channels and args.out_channels:
        model = nn.Sequential(
            *[
                ResidualBlock(
                    in_channels=args.in_channels,
                    out_channels=args.out_channels,
                    index=index,
                )
                for index in range(16)
            ]
        )
        images = torch.randn(1, 64, 64, 64)

        print(model(images).size())

    else:
        raise Exception(
            "Residual channels and output channels are required".capitalize()
        )
