import sys
import argparse
import torch
import torch.nn as nn

sys.path.append("src/")

from utils import params

class FeatureBlock(nn.Module):
    def __init__(self, in_channels = None, out_channels = None, kernel_size=3, stride=2, padding=1):
        super(FeatureBlock, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        self.model = self.feature_block()
        
    def feature_block(self):
        return nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding),
            nn.BatchNorm2d(num_features=self.out_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )       
        
    def forward(self, x):
        if x is not None:
            return self.model(x)
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Features block for netD".title())

    parser.add_argument("--in_channels", type=int, default=1, help="Input channels")
    parser.add_argument("--out_channels", type=int, default=64, help="Output channels")

    args = parser.parse_args()

    if args.in_channels and args.out_channels:
        features_block = FeatureBlock(
            in_channels=args.in_channels, out_channels=args.out_channels
        )

        images = torch.randn(1, 64, 64, 64)

        print(features_block(images).size())

    else:
        raise Exception("Features Block channels and output channels are required".capitalize())