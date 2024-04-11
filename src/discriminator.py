import sys
import argparse
from collections import OrderedDict
import torch
import torch.nn as nn

sys.path.append("src/")

from netD_helpers.input_block import InputBlock
from netD_helpers.features_block import FeatureBlock
from netD_helpers.output_block import OutputBlock

class Discriminator(nn.Module):
    def __init__(self, in_channels = None, out_channels = None):
        super(Discriminator, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filters = out_channels
        
        self.kernel_size = 3
        self.stride = 1
        self.padding = 1
        self.num_repetitive = 7
        self.layers = []
        
        self.input = InputBlock(in_channels=self.in_channels, out_channels=self.out_channels)
        
        for index in range(self.num_repetitive):
            if index%2:
                self.layers.append(FeatureBlock(in_channels = self.out_channels, out_channels=self.out_channels*2))
                
                self.out_channels = self.out_channels*2
            
            else:
                self.layers.append(FeatureBlock(in_channels = self.out_channels, out_channels=self.out_channels))
                
                self.out_channels = self.out_channels
                
        self.features = nn.Sequential(*self.layers)
        
        self.avg_pool = nn.AdaptiveMaxPool2d(output_size=1)
        
        self.output = OutputBlock(in_channels=self.filters, out_channels=1)
        
    def forward(self, x):
        input = self.input(x)
        features = self.features(input)
        output = self.output(self.avg_pool(features))
        
        return output.view(-1)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Discriminator for SRGAN".title())
    parser.add_argument("--in_channels", type=int, default=1, help="Input channels")
    parser.add_argument("--out_channels", type=int, default=64, help="Output channels")
    
    parser.add_argument(
        "--netD", action="store_true", help="Generate a generator".capitalize()
    )

    args = parser.parse_args()

    if args.netD:
        netD = Discriminator(
            in_channels=args.in_channels,
            out_channels=args.out_channels
            )

        images = torch.randn(1, 3, 256, 256)

        print(netD(images).shape)

    else:
        raise Exception("Arguments should be passed".capitalize())