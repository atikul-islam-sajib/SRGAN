import sys
import argparse
from collections import OrderedDict
import torch
import torch.nn as nn

sys.path.append("src/")

from utils import params


class ResidualBlock(nn.Module):
    def __init__(self, in_channels=None, out_channels=None, index=None):
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
