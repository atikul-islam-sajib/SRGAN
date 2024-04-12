import sys
import argparse
from collections import OrderedDict
import torch
import torch.nn as nn

sys.path.append("src/")

from netG_helpers.input_block import InputBlock
from netG_helpers.residual_block import ResidualBlock
from netG_helpers.middle_block import MiddleBlock
from netG_helpers.upsample_block import UpSampleBlock
from netG_helpers.out_block import OutputBlock


class Generator(nn.Module):
    """
    Defines the Generator model for a Super-Resolution GAN (SRGAN). This model aims to upscale low-resolution
    images into high-resolution counterparts. The generator architecture is composed of an input block, multiple
    residual blocks, a middle block, upsample blocks, and an output block.

    The network starts with an InputBlock to process the initial features, followed by a series of ResidualBlocks
    to learn the residual features. A MiddleBlock serves to transition the features before they are upsampled by
    UpSampleBlocks. Finally, an OutputBlock generates the high-resolution output.

    Attributes:
        num_repetitive (int): The number of ResidualBlocks used in the generator.
        input_block (InputBlock): Initial block to process input features.
        residual_block (nn.Sequential): Sequential container of ResidualBlocks for feature learning.
        middle_block (MiddleBlock): Block for feature transition before upsampling.
        up_sample (nn.Sequential): Sequential container of UpSampleBlocks for spatial upsampling of features.
        out_block (OutputBlock): Final block to produce the high-resolution output.
    """

    def __init__(self, in_channels=3, out_channels=64):
        super(Generator, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_repetitive = 16

        self.input_block = InputBlock(
            in_channels=self.in_channels, out_channels=self.out_channels
        )

        self.residual_block = nn.Sequential(
            *[
                ResidualBlock(
                    in_channels=self.out_channels,
                    out_channels=self.out_channels,
                    index=index,
                )
                for index in range(self.num_repetitive)
            ]
        )

        self.middle_block = MiddleBlock(
            in_channels=self.out_channels, out_channels=self.out_channels
        )

        self.up_sample = nn.Sequential(
            *[
                UpSampleBlock(
                    in_channels=self.out_channels,
                    out_channels=self.out_channels * 4,
                    is_first_block=is_first_block,
                    index=index,
                )
                for index, is_first_block in enumerate([True, False])
            ]
        )

        self.out_block = OutputBlock(
            in_channels=self.out_channels, out_channels=self.in_channels
        )

    def forward(self, x):
        """
        Defines the forward pass of the Generator.

        Parameters:
            x (torch.Tensor): The input low-resolution image tensor.

        Raises:
            Exception: If the input tensor x is None, indicating that the implementation of the generator is incomplete.

        Returns:
            torch.Tensor: The output high-resolution image tensor.
        """
        if x is not None:
            input = self.input_block(x)
            residual = self.residual_block(input)
            middle = self.middle_block(residual, input)
            upsample = self.up_sample(middle)
            output = self.out_block(upsample)

            return output

        else:
            raise Exception("Generator not implemented".capitalize())

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
    parser = argparse.ArgumentParser(description="Generator for SRGAN".title())

    parser.add_argument("--in_channels", type=int, default=64, help="Input channels")
    parser.add_argument("--out_channels", type=int, default=64, help="Output channels")

    parser.add_argument(
        "--netG", action="store_true", help="Generate a generator".capitalize()
    )

    args = parser.parse_args()

    if args.netG:
        netG = Generator(in_channels=args.in_channels, out_channels=args.out_channels)

        images = torch.randn(64, 3, 64, 64)

        print(netG(images).shape)

        print(
            "Total params of the Generator model is # {}".format(
                Generator.total_params(model=netG)
            )
        )

    else:
        raise Exception("Arguments should be passed".capitalize())
