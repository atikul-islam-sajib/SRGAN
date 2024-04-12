import sys
import argparse
import torch
import torch.nn as nn
import torchvision.models as models

sys.path.append("src/")


class VGG16(nn.Module):
    """
    A module configured to use either VGG16 or VGG19 architecture up to a specified layer for feature extraction.
    This class allows for the use of pretrained models with an option to freeze weights, making it suitable for
    feature extraction without further training the network.

    Attributes:
        pretrained (bool): Specifies whether to load the model with pretrained weights.
        freeze_weights (bool): Specifies whether to freeze the model weights to prevent training during operations.
        is_vgg16 (bool): Determines which model to use; VGG16 if True, otherwise VGG19 by default.
        model (nn.Module): The actual model truncated for feature extraction, either VGG16 or VGG19.
        num_layers (int): Number of layers to retain from the original architecture for feature extraction.
    """

    def __init__(self, pretrained=True, freeze_weights=False, is_vgg16=False):
        """
        Initializes the VGG16 class either with VGG16 or VGG19 architecture truncated to include only the
        convolutional layers necessary for feature extraction up to the specified layer count.

        Parameters:
            pretrained (bool, optional): If True, initializes the model with pretrained weights. Defaults to True.
            freeze_weights (bool, optional): If True, freezes the model weights to prevent updates during training. Defaults to True.
            is_vgg16 (bool, optional): If True, uses the VGG16 architecture; otherwise, uses VGG19. Defaults to False.
        """
        super(VGG16, self).__init__()

        self.name = "Feature Extractor"

        self.pretrained = pretrained
        self.freeze_weights = freeze_weights
        self.is_vgg16 = is_vgg16
        self.num_layers = 18

        if self.is_vgg16:
            self.model = models.vgg16(pretrained=True)

            for params in self.model.parameters():
                params.requires_grad = False

            self.num_layers = 14

        else:
            self.model = models.vgg19(pretrained=True)

            for params in self.model.parameters():
                params.requires_grad = False

        self.model = nn.Sequential(
            *list(self.model.features.children())[: self.num_layers]
        )

    def forward(self, x):
        """
        Defines the forward pass through either the VGG16 or VGG19 model.

        Parameters:
            x (torch.Tensor): The input tensor for feature extraction.

        Returns:
            torch.Tensor: The tensor containing features extracted by the model up to the specified layer.

        Raises:
            Exception: If no input tensor is provided.
        """
        if x is not None:
            return self.model(x)
        else:
            raise Exception("No input is provided".capitalize())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Feature Extractor using VGG19".capitalize()
    )

    parser.add_argument(
        "--pretrained",
        type=bool,
        default=False,
        help="Use pretrained weights".capitalize(),
    )
    parser.add_argument(
        "--freeze_weights",
        type=bool,
        default=False,
        help="Freeze the weights".capitalize(),
    )
    parser.add_argument(
        "--is_vgg16",
        type=bool,
        default=False,
        help="Use VGG16".capitalize(),
    )

    args = parser.parse_args()

    if args.pretrained and args.freeze_weights and args.is_vgg16:
        net = VGG16(
            pretrained=args.pretrained,
            freeze_weights=args.freeze_weights,
            is_vgg16=args.is_vgg16,
        )

        images = torch.randn(1, 3, 256, 256)

        print(net(images).size())

    else:
        raise Exception("Arguments not provided".capitalize())
