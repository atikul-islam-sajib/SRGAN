import sys
import os
import argparse

sys.path.append("src/")

from dataloader import Loader
from generator import Generator
from discriminator import Discriminator
from trainer import Trainer
from charts import Test


def cli():
    parser = argparse.ArgumentParser(description="Data Loader for SR-GAN".title())
    parser.add_argument(
        "--image_path", type=str, help="Path to the image for SR-GAN".capitalize()
    )
    parser.add_argument(
        "--batch_size", type=int, help="Batch size for the dataloader".capitalize()
    )
    parser.add_argument(
        "--image_size", type=int, help="Image size for the dataloader".capitalize()
    )
    parser.add_argument(
        "--is_sub_samples",
        type=bool,
        default=False,
        help="Image size for the dataloader".capitalize(),
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of epochs".capitalize()
    )
    parser.add_argument(
        "--lr", type=float, default=0.0002, help="Learning rate".capitalize()
    )
    parser.add_argument(
        "--content_loss",
        type=float,
        default=1e-3,
        help="Content loss weight".capitalize(),
    )
    parser.add_argument(
        "--is_l1",
        type=bool,
        default=False,
        help="Use L1 loss instead of L2 loss".capitalize(),
    )
    parser.add_argument(
        "--is_l2",
        type=bool,
        default=False,
        help="Use L2 loss instead of L1 loss".capitalize(),
    )
    parser.add_argument(
        "--is_elastic_net",
        type=bool,
        default=False,
        help="Use Elastic Net loss instead of L1 and L2 loss".capitalize(),
    )
    parser.add_argument(
        "--is_lr_scheduler",
        type=bool,
        default=False,
        help="Use learning rate scheduler".capitalize(),
    )
    parser.add_argument(
        "--is_weight_init",
        type=bool,
        default=False,
        help="Use weight initialization".capitalize(),
    )
    parser.add_argument(
        "--is_weight_clip",
        type=float,
        default=0.01,
        help="Use weight Clipping in netG".capitalize(),
    )
    parser.add_argument(
        "--is_display",
        type=bool,
        default=False,
        help="Display detailed loss information".capitalize(),
    )
    parser.add_argument("--device", type=str, default="mps", help="Device".capitalize())
    parser.add_argument(
        "--adam", type=bool, default=True, help="Use Adam optimizer".capitalize()
    )
    parser.add_argument(
        "--SGD", type=bool, default=False, help="Use SGD optimizer".capitalize()
    )
    parser.add_argument(
        "--beta1", type=float, default=0.5, help="Beta1 parameter".capitalize()
    )
    parser.add_argument(
        "--train", action="store_true", help="Train the model".capitalize()
    )
    parser.add_argument(
        "--model", type=str, default=None, help="Path to the best model".capitalize()
    )
    parser.add_argument(
        "--test", action="store_true", help="Test the model".capitalize()
    )

    args = parser.parse_args()

    if args.train:
        loader = Loader(
            image_path=args.image_path,
            batch_size=args.batch_size,
            image_size=args.image_size,
            is_sub_samples=args.is_sub_samples,
        )

        loader.unzip_folder()
        loader.create_dataloader()

        trainer = Trainer(
            epochs=args.epochs,
            lr=args.lr,
            content_loss=args.content_loss,
            device=args.device,
            display=args.device,
            adam=args.adam,
            SGD=args.SGD,
            beta1=args.beta1,
            is_l1=args.is_l1,
            is_l2=args.is_l2,
            is_elastic_net=args.is_elastic_net,
            is_lr_scheduler=args.is_lr_scheduler,
            is_weight_clip=args.is_weight_clip,
            is_weight_init=args.is_weight_init,
        )

        trainer.train()

    elif args.test:
        test = Test(device=args.device, model=args.model)
        test.plot()


if __name__ == "__main__":
    cli()
