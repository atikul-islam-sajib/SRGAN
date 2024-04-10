import sys
import argparse
from collections import OrderedDict
import torch
import torch.nn as nn

sys.path.append("src/")

from utils import params


class MiddleBlock(nn.Module):
    """
    A neural network module designed as a middle block for a generator network (netG), implementing a convolutional
    layer followed by batch normalization.

    This module is structured to support skip connections, enabling the addition of input with another tensor
    before passing it through subsequent layers, thus allowing for features reusability and deeper network training
    without vanishing gradient issues.

    Attributes:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        kernel_size (int): Kernel size for the convolutional layer.
        stride (int): Stride for the convolutional layer.
        padding (int): Padding for the convolutional layer.
        model (nn.Sequential): Sequential model comprising convolutional and batch normalization layers.

    Examples:
        >>> middle_block = MiddleBlock(in_channels=64, out_channels=128)
        >>> images = torch.randn(1, 64, 64, 64)
        >>> skip_info = torch.randn(1, 128, 64, 64)
        >>> output = middle_block(images, skip_info)
        >>> print(output.size())
        torch.Size([1, 128, 64, 64])
    """
    def __init__(self, in_channels=None, out_channels=None):
        """
        Initializes the middle block with given input and output channels, setting up the model architecture.
        
        Parameters:
            in_channels (int, optional): The number of channels in the input tensor. Defaults to None.
            out_channels (int, optional): The number of channels produced by the convolution. Defaults to None.
        """
        super(MiddleBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.kernel_size = params()["netG"]["block"]["kernel_size"]
        self.stride = params()["netG"]["block"]["stride"]
        self.padding = params()["netG"]["block"]["padding"]

        try:
            self.model = self.middle_block()
        except Exception as _:
            print("Middle block not implemented")

    def middle_block(self):
        """
        Constructs the layers for the middle block, including a convolutional layer and a batch normalization layer.
        
        Returns:
            nn.Sequential: A sequential model containing the defined layers.
        """
        layers = OrderedDict()

        layers["conv"] = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
        )

        layers["batchnorm"] = nn.BatchNorm2d(num_features=self.out_channels)

        return nn.Sequential(layers)

    def forward(self, x=None, skip_info=None):
        """
        Forward pass through the middle block with an option for a skip connection.
        
        Parameters:
            x (torch.Tensor, optional): The input tensor to the block. Defaults to None.
            skip_info (torch.Tensor, optional): An additional tensor for skip connections. Defaults to None.
        
        Raises:
            Exception: If either `x` or `skip_info` is None, indicating incomplete implementation.
        
        Returns:
            torch.Tensor: The output tensor after passing through the block and adding `skip_info`.
        """
        if (x is not None) and (skip_info is not None):
            return self.model(x) + skip_info
        else:
            raise Exception("Middle block not implemented".capitalize())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in_channels", type=int, default=1, help="Define the in_channels".capitalize()
    )
    parser.add_argument(
        "--out_channels",
        type=int,
        default=1,
        help="Define the in_channels".capitalize(),
    )

    args = parser.parse_args()

    if args.in_channels and args.out_channels:
        model = MiddleBlock(
            in_channels=args.in_channels, out_channels=args.out_channels
        )
        images = torch.randn(1, 64, 64, 64)
        input = torch.randn(1, 64, 64, 64)

        print(model(images, input).size())

    else:
        raise Exception(
            "Middle Block channels and output channels are required".capitalize()
        )
