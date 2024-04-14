import sys
import os
import yaml
import argparse

sys.path.append("src/")

from dataloader import Loader
from trainer import Trainer
from charts import Test
from inference import Inference


def saved_params(**kwargs):
    """
    Saves provided parameters to a YAML file. This function takes any number of keyword arguments and writes them
    to a YAML file, making it useful for saving configuration settings or model parameters after training.

    The data is saved in a human-readable format, which can be easily loaded later for review or reuse.

    Parameters:
        **kwargs (dict): Arbitrary keyword arguments that represent the parameters to be saved. This could include
                         model hyperparameters, training settings, or any other configuration details.

    Side Effects:
        Writes to a file named 'trained_params.yml' in the current working directory. If the file already exists,
        it will be overwritten.

    Raises:
        IOError: If the file could not be opened, written, or closed due to I/O errors.

    Example Usage:
        saved_params(learning_rate=0.001, epochs=100, optimizer='Adam')
    """
    with open("./trained_params.yml", "w") as file:
        yaml.safe_dump(kwargs, file, default_flow_style=False)


def cli():
    """
    Command-line interface function for managing the data loading, training, and testing processes for a Super-Resolution
    GAN (SR-GAN). It parses command-line arguments to configure and initiate various operational modes including data
    preparation, model training, and model testing.

    The CLI provides options to specify image paths, training parameters, and whether to train or test the model,
    along with settings for optimizer types, learning rates, loss types, and device configuration.

    Command-Line Arguments:
        --image_path (str): Path to the image dataset.
        --batch_size (int): Number of images per batch during training.
        --image_size (int): Size to which the images will be resized.
        --is_sub_samples (bool): Whether to create subsamples of the dataset.
        --epochs (int): Number of training epochs.
        --lr (float): Learning rate for the optimizer.
        --content_loss (float): Weight of the content loss in the loss function.
        --is_l1 (bool): Whether to use L1 loss.
        --is_l2 (bool): Whether to use L2 loss.
        --is_elastic_net (bool): Whether to use Elastic Net loss.
        --is_lr_scheduler (bool): Whether to apply a learning rate scheduler.
        --is_weight_init (bool): Whether to initialize weights before training.
        --is_weight_clip (float): Magnitude of weight clipping in netG.
        --is_display (bool): Whether to display detailed training progress.
        --device (str): Computation device to use ('cuda', 'mps', 'cpu').
        --adam (bool): Whether to use the Adam optimizer.
        --SGD (bool): Whether to use Stochastic Gradient Descent.
        --beta1 (float): Beta1 hyperparameter for Adam optimizer.
        --train (action): Flag to indicate training mode.
        --model (str): Path to a saved model for testing.
        --test (action): Flag to indicate testing mode.

    Raises:
        Exception: Handles general exceptions related to file paths or operational issues, providing a message
                   indicating the nature of the error.
    """
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
        type=bool,
        default=False,
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
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to the image for SRGAN".capitalize(),
    )
    parser.add_argument(
        "--single", action="store_true", help="Single image inference".capitalize()
    )
    parser.add_argument(
        "--batch", action="store_true", help="Batch image inference".capitalize()
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
            display=args.is_display,
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

        try:
            params = {
                "dataloader": {
                    "batch_size": args.batch_size,
                    "image_size": args.image_size,
                    "is_sub_samples": args.is_sub_samples,
                },
                "trainer": {
                    "epochs": args.epochs,
                    "lr": args.lr,
                    "content_loss": args.content_loss,
                    "device": args.device,
                    "display": args.is_display,
                    "adam": args.adam,
                    "SGD": args.SGD,
                    "beta1": args.beta1,
                    "is_l1": args.is_l1,
                    "is_l2": args.is_l2,
                    "is_elastic_net": args.is_elastic_net,
                    "is_lr_scheduler": args.is_lr_scheduler,
                    "is_weight_clip": args.is_weight_clip,
                    "is_weight_init": args.is_weight_init,
                },
            }

            saved_params(params=params)

        except Exception as e:
            print("Exception occurred: {}".format(e))

    elif args.test:
        test = Test(device=args.device, model=args.model)
        test.plot()

    elif args.single:
        inference = Inference(image=args.image, model=args.model, device=args.device)
        inference.srgan_single()

    elif args.batch:
        inference = Inference(image=args.image, model=args.model, device=args.device)
        inference.srgan_batch()


if __name__ == "__main__":
    cli()
