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
    def __init__(self):
        super(Generator, self).__init__()

        self.num_repetitive = 16

        self.input_block = InputBlock(in_channels=3, out_channels=64)

        self.residual_block = nn.Sequential(
            *[
                ResidualBlock(in_channels=64, out_channels=64, index=index)
                for index in range(self.num_repetitive)
            ]
        )

        self.middle_block = MiddleBlock(in_channels=64, out_channels=64)

        self.up_sample = nn.Sequential(
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

        self.out_block = OutputBlock(in_channels=64, out_channels=3)

    def forward(self, x):
        if x is not None:
            input = self.input_block(x)
            residual = self.residual_block(input)
            middle = self.middle_block(residual, input)
            upsample = self.up_sample(middle)
            output = self.out_block(upsample)

            return output

        else:
            raise Exception("Generator not implemented".capitalize())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generator for SRGAN".title())
    parser.add_argument(
        "--netG", action="store_true", help="Generate a generator".capitalize()
    )

    args = parser.parse_args()

    if args.netG:
        netG = Generator()

        images = torch.randn(64, 3, 64, 64)

        print(netG(images).shape)

    else:
        raise Exception("Arguments should be passed".capitalize())
