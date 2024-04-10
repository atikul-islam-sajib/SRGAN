import sys
import argparse
from collections import OrderedDict
import torch
import torch.nn as nn

sys.path.append("src/")

from utils import params


class OutputBlock(nn.Module):
    """
    Defines the output block for a neural network generator (netG), applying a convolutional layer followed by
    a Tanh activation to generate the final output. This block is designed to convert the feature maps from
    the preceding layers to the desired number of output channels, typically corresponding to the image's
    color channels.

    Attributes:
        in_channels (int): Number of input channels to the block.
        out_channels (int): Desired number of output channels.
        kernel_size (int): Kernel size for the convolutional layer.
        stride (int): Stride for the convolutional layer.
        padding (int): Padding for the convolutional layer.
        model (nn.Sequential): Sequential model containing the convolutional and Tanh activation layers.

    Examples:
        >>> output_block = OutputBlock(in_channels=64, out_channels=3)
        >>> images = torch.randn(1, 64, 256, 256)
        >>> output = output_block(images)
        >>> print(output.size())
        torch.Size([1, 3, 256, 256])
    """
    def __init__(self, in_channels=None, out_channels=None):
        """
        Initializes the output block with specified input and output channels, constructing the model architecture.

        Parameters:
            in_channels (int, optional): The number of channels in the input tensor. Defaults to None.
            out_channels (int, optional): The number of channels in the output tensor. Defaults to None.
        """
        super(OutputBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = params()["netG"]["input"]["kernel_size"]
        self.stride = params()["netG"]["input"]["stride"]
        self.padding = params()["netG"]["input"]["padding"]

        try:
            self.model = self.output_block()
        except Exception as _:
            print("Output block not implemented".capitalize())

    def output_block(self):
        """
        Constructs the output block's layers, including a convolutional layer followed by a Tanh activation layer.
        
        Returns:
            nn.Sequential: The sequential container holding the defined layers.
        """
        layers = OrderedDict()

        layers["conv"] = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
        )
        layers["tanh"] = nn.Tanh()

        return nn.Sequential(layers)

    def forward(self, x):
        """
        Forward pass through the output block.

        Parameters:
            x (torch.Tensor): The input tensor to the block.

        Raises:
            Exception: If the input tensor x is None, indicating incomplete implementation.
        
        Returns:
            torch.Tensor: The output tensor after processing by the block.
        """
        if x is not None:
            return self.model(x)
        else:
            raise Exception("Output block not implemented".capitalize())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in_channels",
        type=int,
        default=64,
        help="Define the in_channels".capitalize(),
    )
    parser.add_argument(
        "--out_channels",
        type=int,
        default=3,
        help="Define the out_channels".capitalize(),
    )

    args = parser.parse_args()

    if args.in_channels and args.out_channels:
        out = OutputBlock(in_channels=args.in_channels, out_channels=args.out_channels)

        images = torch.randn(1, 64, 256, 256)

        print(out(images).size())
    else:
        raise Exception(
            "OutBlock channels and output channels are required".capitalize()
        )
