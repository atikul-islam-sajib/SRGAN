import sys
import argparse
from collections import OrderedDict
import torch
import torch.nn as nn

sys.path.append("src/")

from utils import params


class InputBlock(nn.Module):
    def __init__(self, in_channels=None, out_channels=None):
        super(InputBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.kernel_size = params()["netG"]["input"]["kernel_size"]
        self.stride = params()["netG"]["input"]["stride"]
        self.padding = params()["netG"]["input"]["padding"]

        try:
            self.model = self.input_block()
        except Exception as e:
            print("Input block not implemented")

    def input_block(self):
        layers = OrderedDict()

        layers["conv"] = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
        )
        layers["PReLU"] = nn.PReLU()

        return nn.Sequential(layers)

    def forward(self, x=None):
        if x is not None:
            return self.model(x)
        else:
            raise Exception("Input block not implemented")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Input block for netG".title())

    parser.add_argument("--in_channels", type=int, default=1, help="Input channels")
    parser.add_argument("--out_channels", type=int, default=64, help="Output channels")

    args = parser.parse_args()

    if args.in_channels and args.out_channels:
        input_block = InputBlock(
            in_channels=args.in_channels, out_channels=args.out_channels
        )

        images = torch.randn(1, 3, 64, 64)

        print(input_block(images).size())

    else:
        raise Exception("Input channels and output channels are required".capitalize())
