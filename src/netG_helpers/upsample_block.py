import sys
import argparse
from collections import OrderedDict
import torch
import torch.nn as nn

sys.path.append("src/")

from utils import params


class UpSampleBlock(nn.Module):
    """
    Represents an upsample block within a neural network generator (netG), which increases the spatial dimensions
    of the input feature maps. This is commonly used in generator architectures for tasks like super-resolution.

    The block applies a convolution followed by a PixelShuffle operation to upsample the input. Optionally, if it is
    the first upsample block in the sequence, a PReLU activation function is applied for non-linearity.

    Attributes:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels, which should be a multiple of the square of the upscale factor.
        is_first_block (bool): Indicates if this is the first upsample block in the network, determining if a PReLU activation is used.
        index (int): A unique identifier for the block, used for naming the layers.
        kernel_size (int): The kernel size for the convolutional layer.
        stride (int): The stride for the convolutional layer.
        padding (int): The padding for the convolutional layer.
        factor (int): The upscale factor for the PixelShuffle operation.
        model (nn.Sequential): The sequential model containing the layers of the block.

    Examples:
        >>> upsample_block = UpSampleBlock(in_channels=64, out_channels=256, is_first_block=True, index=0)
        >>> images = torch.randn(1, 64, 64, 64)
        >>> output = upsample_block(images)
        >>> print(output.size())
        torch.Size([1, 64, 128, 128])
    """
    def __init__(
        self, in_channels=None, out_channels=None, is_first_block=False, index=None
    ):
        """
        Initializes the UpSampleBlock with the specified configuration.

        Parameters:
            in_channels (int, optional): The number of input channels. Defaults to None.
            out_channels (int, optional): The number of output channels; should be a multiple of the square of the upscale factor. Defaults to None.
            is_first_block (bool, optional): If True, includes a PReLU activation layer. Defaults to False.
            index (int, optional): A unique identifier for the layers within the block. Defaults to None.
        """
        super(UpSampleBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.is_first_block = is_first_block
        self.index = index

        self.kernel_size = params()["netG"]["block"]["kernel_size"]
        self.stride = params()["netG"]["block"]["stride"]
        self.padding = params()["netG"]["block"]["padding"]
        self.factor = params()["netG"]["block"]["upscale_factor"]

        try:
            self.model = self.up_sample_block()
        except Exception as _:
            print("Up sample block not implemented".capitalize())

    def up_sample_block(self):
        """
        Constructs the upsample block's layers, including a convolutional layer, a PixelShuffle operation, 
        and optionally a PReLU activation layer if it is the first block.

        Returns:
            nn.Sequential: The sequential model comprising the defined layers.
        """
        layers = OrderedDict()
        layers["conv{}".format(self.index)] = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
        )
        layers["pixel_shuffle{}".format(self.index)] = nn.PixelShuffle(
            upscale_factor=self.factor
        )

        if self.is_first_block:
            layers["PReLU"] = nn.PReLU()

        return nn.Sequential(layers)

    def forward(self, x):
        """
        Defines the forward pass through the upsample block.

        Parameters:
            x (torch.Tensor): The input tensor to the block.

        Raises:
            Exception: If the input tensor x is None, indicating that the implementation of the block is incomplete.

        Returns:
            torch.Tensor: The output tensor after upsampling.
        """
        if x is not None:
            return self.model(x)
        else:
            raise Exception("Up sample block not implemented".capitalize())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upsample block for netG".title())

    parser.add_argument(
        "--in_channels",
        type=int,
        default=64,
        help="Input channels for the block",
    )

    parser.add_argument(
        "--out_channels",
        type=int,
        default=256,
        help="Output channels for the block",
    )

    args = parser.parse_args()

    if args.in_channels and args.out_channels:

        up_sample = nn.Sequential(
            *[
                UpSampleBlock(
                    in_channels=64,
                    out_channels=256,
                    is_first_block=is_first_block,
                    index=index,
                )
                for index, is_first_block in enumerate([True, False])
            ]
        )

        images = torch.randn(1, 64, 64, 64)

        print(up_sample(images).size())

    else:
        raise Exception(
            "UpSample channels and output channels are required".capitalize()
        )
