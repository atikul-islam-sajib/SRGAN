import sys
import argparse
import torch
import torch.nn as nn
import torchvision.models as models

sys.path.append("src/")


class VGG16(nn.Module):
    def __init__(self, pretrained=True, freeze_weights=False):
        self.name = "Feature Extractor"

        self.pretrained = pretrained
        self.freeze_weights = freeze_weights

        self.model = models.vgg19(pretrained=True)

        for params in self.model.params:
            params.requires_grad = False

        self.model = nn.Sequential(*list(self.model.features.children()[:18]))

    def forward(self, x):
        if x is not None:
            return self.model(x)
        else:
            raise Exception("No input is provided".capitalize())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Feature Extractor using VGG19".capitalize()
    )

    parser.add_argument(
        "--pretrained", action="store_true", help="Use pretrained weights".capitalize()
    )
    parser.add_argument(
        "--freeze_weights",
        action="store_true",
        help="Freeze the weights".capitalize(),
    )

    args = parser.parse_args()

    if args.pretrained and args.freeze_weights:
        net = VGG16(pretrained=args.pretrained, freeze_weights=args.freeze_weights)

        images = torch.randn(1, 3, 256, 256)

        print(net(images).size())

    else:
        raise Exception("Arguments not provided".capitalize())
