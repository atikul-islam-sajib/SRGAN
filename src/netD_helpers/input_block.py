import sys
import argparse
from collections import OrderedDict
import torch
import torch.nn as nn

sys.path.append("src/")

from utils import params

class InputBlock(nn.Module):
    def __init__(self, in_channels = None, out_channels = None):
        super(InputBlock, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.kernel_size = params()["netD"]["kernel_size"]
        self.stride = params()["netD"]["stride"]
        self.padding = params()["netD"]["padding"]
        
        self.model = self.input_block()
        
    def input_block(self):
        return nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, self.kernel_size, self.stride, self.padding),
            nn.BatchNorm2d(num_features=self.out_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        
    def forward(self, x):
        if x is not None:
            return self.model(x)
        
        else:
            raise Exception("Input channels and output channels are required".capitalize())
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Input block for netD".title())

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