import sys
import argparse
import torch
import torch.nn as nn

sys.path.append("src/")

from utils import params

class InputBlock(nn.Module):
    """
    Represents the initial block of the discriminator network (netD), designed to process input images using
    a sequence of convolutional and activation layers. This block primarily focuses on capturing the basic patterns
    from the input images and preparing the features for deeper analysis in subsequent layers.

    The block employs LeakyReLU as the activation function to introduce non-linearity, allowing the model to learn
    more complex patterns. Additionally, batch normalization is applied after the second convolutional layer to
    stabilize the learning process by normalizing the features.

    Attributes:
        in_channels (int): The number of channels in the input image.
        out_channels (int): The number of filters in the convolutional layers, defining the output depth.
        kernel_size (int): The size of the kernel used in the convolutional layers.
        stride (int): The stride of the convolutional operations, affecting the downsampling rate.
        padding (int): The padding applied to the input tensor before convolution.
        model (nn.Sequential): The sequential container comprising the layers of the block.

    Examples:
        >>> input_block = InputBlock(in_channels=3, out_channels=64)
        >>> images = torch.randn(1, 3, 64, 64)
        >>> output = input_block(images)
        >>> print(output.size())
        torch.Size([1, 64, 16, 16])
    """
    def __init__(self, in_channels = None, out_channels = None):
        """
        Initializes the InputBlock with the specified number of input and output channels, along with convolutional
        parameters fetched from the network parameters (`params()`).

        Parameters:
            in_channels (int, optional): The number of channels in the input images. Defaults to None.
            out_channels (int, optional): The desired number of channels in the output tensor. Defaults to None.
        """
        super(InputBlock, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.kernel_size = params()["netD"]["kernel_size"]
        self.stride = params()["netD"]["stride"]
        self.padding = params()["netD"]["padding"]
        
        self.model = self.input_block()
        
    def input_block(self):
        """
        Constructs the input block, including two convolutional layers with a LeakyReLU activation function after
        each and batch normalization following the second convolutional layer.

        Returns:
            nn.Sequential: The sequential model containing the defined layers.
        """
        return nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, self.kernel_size, self.stride, self.padding),
            nn.BatchNorm2d(num_features=self.out_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        
    def forward(self, x):
        """
        Defines the forward pass of the InputBlock.

        Parameters:
            x (torch.Tensor): The input tensor to the block.

        Returns:
            torch.Tensor: The output tensor after processing by the block.

        Raises:
            Exception: If the input tensor x is None, indicating that the input has not been provided.
        """
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