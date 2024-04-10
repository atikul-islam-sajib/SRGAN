import sys
import argparse
from collections import OrderedDict
import torch
import torch.nn as nn

sys.path.append("src/")

from utils import params


class InputBlock(nn.Module):
    """
    A module representing the input block of a neural network generator (netG).
    
    This block initializes with the input and output channels and sets up a convolutional layer 
    followed by a PReLU activation layer based on parameters specified in the `params` function.
    
    Attributes:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        kernel_size (int): Kernel size for the convolutional layer.
        stride (int): Stride for the convolutional layer.
        padding (int): Padding for the convolutional layer.
        model (nn.Sequential): Sequential model comprising the convolutional and PReLU layers.
    
    Examples:
        >>> input_block = InputBlock(in_channels=3, out_channels=64)
        >>> images = torch.randn(1, 3, 64, 64)
        >>> output = input_block(images)
        >>> print(output.size())
        torch.Size([1, 64, 64, 64])
    """
    def __init__(self, in_channels=None, out_channels=None):
        """
        Initializes the input block with the given input and output channels.
        
        Parameters:
            in_channels (int, optional): The number of channels in the input image. Defaults to None.
            out_channels (int, optional): The number of channels produced by the convolution. Defaults to None.
        """
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
        """
        Defines the layers of the input block.
        
        This method constructs an OrderedDict of layers comprising a convolutional layer followed by a PReLU activation layer.
        
        Returns:
            nn.Sequential: A sequential container of the layers defined.
        """
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
        """
        Defines the forward pass of the input block.
        
        Parameters:
            x (torch.Tensor, optional): The input tensor. Defaults to None.
        
        Raises:
            Exception: If the input tensor x is None.
        
        Returns:
            torch.Tensor: The output tensor after passing through the model.
        """
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
