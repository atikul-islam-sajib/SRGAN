import sys
import argparse
import torch
import torch.nn as nn

sys.path.append("src/")

class OutputBlock(nn.Module):
    """
    Represents the final output block of the discriminator network (netD), designed to consolidate features extracted
    by previous layers and output a single value that signifies whether the input is considered real or fake.

    This block progressively increases the depth of the input features before applying a final convolution that reduces
    the output to a single channel. A Tanh activation function is applied at the end to normalize the output to the range
    [-1, 1], which is a common practice for GAN discriminators.

    Attributes:
        in_channels (int): The number of input channels to the block.
        out_channels (int): The base number of output channels, used for scaling within the block.
        kernel_size (int): The size of the kernel used in the convolutional layers.
        model (nn.Sequential): The sequential model comprising the layers of the block.

    Examples:
        >>> output_block = OutputBlock(in_channels=64, out_channels=64)
        >>> images = torch.randn(1, 512, 4, 4)  # Example input size
        >>> output = output_block(images)
        >>> print(output.size())
        torch.Size([1, 1, 1, 1])
    """
    def __init__(self, in_channels = None, out_channels=None):
        """
        Initializes the OutputBlock with the specified number of input and output channels. The `kernel_size` is
        set to match the `out_channels`, which may need adjustment based on the network design and input dimensions.

        Parameters:
            in_channels (int, optional): The number of channels in the input tensor. Defaults to None.
            out_channels (int, optional): The base number of output channels for scaling purposes. Defaults to None.
        """
        super(OutputBlock, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = out_channels
        
        self.model = self.output_block()
        
    def output_block(self):
        """
        Constructs the output block's layers, including two convolutional layers with LeakyReLU activation,
        and a final Tanh activation layer.

        Returns:
            nn.Sequential: The sequential model containing the defined layers.
        """
        return nn.Sequential(
            nn.Conv2d(self.in_channels*8, self.in_channels*16, self.kernel_size),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(self.in_channels*16, 1, self.kernel_size),
            nn.Tanh() 
        )
        
    def forward(self, x):
        """
        Defines the forward pass of the OutputBlock.

        Parameters:
            x (torch.Tensor): The input tensor to the block.

        Returns:
            torch.Tensor: The output tensor after processing by the block.

        Raises:
            Exception: If the input tensor x is None, indicating that the input has not been provided.
        """
        if x is not None:
            return self.model(x)
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Output block for netD".title())

    parser.add_argument("--in_channels", type=int, default=64, help="Input channels")
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