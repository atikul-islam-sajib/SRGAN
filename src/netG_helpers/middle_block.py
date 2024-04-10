import sys
import argparse
from collections import OrderedDict
import torch
import torch.nn as nn

sys.path.append("src/")

from utils import params


class MiddleBlock(nn.Module):
    def __init__(self, in_channels=None, out_channels=None):
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
