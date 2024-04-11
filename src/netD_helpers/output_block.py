import sys
import argparse
import torch
import torch.nn as nn

sys.path.append("src/")

class OutputBlock(nn.Module):
    def __init__(self, in_channels = None, out_channels=None):
        super(OutputBlock, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = out_channels
        
        self.model = self.output_block()
        
    def output_block(self):
        return nn.Sequential(
            nn.Conv2d(self.in_channels*8, self.in_channels*16, self.kernel_size),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(self.in_channels*16, 1, self.kernel_size),
            nn.Tanh() 
        )
        
    def forward(self, x):
        if x is not None:
            return self.model(x)
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Output block for netD".title())

    parser.add_argument("--in_channels", type=int, default=1, help="Input channels")
    parser.add_argument("--out_channels", type=int, default=64, help="Output channels")

    args = parser.parse_args()

    if args.in_channels and args.out_channels:
        out_block = OutputBlock(
            in_channels=args.in_channels, out_channels=args.out_channels
        )

        images = torch.randn(1, 64, 64, 64)

        print(out_block(images).size())

    else:
        raise Exception("Output Block channels and output channels are required".capitalize())