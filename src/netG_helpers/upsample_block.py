import sys
import argparse
from collections import OrderedDict
import torch
import torch.nn as nn

sys.path.append("src/")

from utils import params


class UpSampleBlock(nn.Module):
    def __init__(
        self, in_channels=None, out_channels=None, is_first_block=False, index=None
    ):
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
