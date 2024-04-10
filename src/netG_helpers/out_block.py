import sys
import argparse
from collections import OrderedDict
import torch
import torch.nn as nn

sys.path.append("src/")

from utils import params


class OutputBlock(nn.Module):
    def __init__(self, in_channels=None, out_channels=None):
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
