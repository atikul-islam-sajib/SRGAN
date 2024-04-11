import sys
import argparse
import torch
import torch.nn as nn

sys.path.append("src/")

from utils import params

class FeatureBlock(nn.Module):
    """
    Defines a feature block for the discriminator network (netD) in a GAN architecture. This block is designed to
    extract features from input images using a convolutional layer, followed by batch normalization and a LeakyReLU
    activation for non-linearity.

    This type of block is typically used in the early layers of the discriminator to progressively downsample the
    input image and increase the channel depth, enabling the network to learn rich feature representations.

    Attributes:
        in_channels (int): The number of channels in the input tensor.
        out_channels (int): The number of filters in the convolutional layer, determining the output tensor's depth.
        kernel_size (int): The size of the convolutional kernel.
        stride (int): The stride of the convolution operation, controlling the downsampling factor.
        padding (int): The padding applied to the input tensor before the convolution operation.
        model (nn.Sequential): The sequential model comprising the convolutional, batch normalization, and LeakyReLU layers.

    Examples:
        >>> feature_block = FeatureBlock(in_channels=3, out_channels=64)
        >>> images = torch.randn(1, 3, 256, 256)
        >>> output = feature_block(images)
        >>> print(output.size())
        torch.Size([1, 64, 128, 128])
    """
    def __init__(self, in_channels = None, out_channels = None, kernel_size=3, stride=2, padding=1):
        """
        Initializes the FeatureBlock with the given parameters for the convolutional layer and its subsequent
        normalization and activation layers.

        Parameters:
            in_channels (int, optional): The number of channels in the input image. Defaults to None.
            out_channels (int, optional): The number of channels produced by the convolution. Defaults to None.
            kernel_size (int, optional): The size of the kernel in the convolutional layer. Defaults to 3.
            stride (int, optional): The stride of the convolutional operation. Defaults to 2.
            padding (int, optional): The padding added to the input tensor along its edges. Defaults to 1.
        """
        super(FeatureBlock, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        self.model = self.feature_block()
        
    def feature_block(self):
        """
        Constructs the feature block's layers, including a convolutional layer, batch normalization, and LeakyReLU activation.

        Returns:
            nn.Sequential: The sequential container holding the defined layers.
        """
        return nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding),
            nn.BatchNorm2d(num_features=self.out_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )       
        
    def forward(self, x):
        """
        Defines the forward pass of the FeatureBlock.

        Parameters:
            x (torch.Tensor): The input tensor to the block.

        Returns:
            torch.Tensor: The output tensor after processing by the block.
        """
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