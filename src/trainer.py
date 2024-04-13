import torch
import torch.nn as nn


class Trainer(nn.Module):
    def __init__(
        self,
        epochs=100,
        lr=0.0002,
        device="mps",
        adam=True,
        SDG=False,
        beta1=0.5,
        is_l1=False,
        is_l2=False,
        is_elastic_net=False,
        is_lr_scheduler=False,
        display=True,
    ):
        self.epochs = epochs
        self.lr = lr
        self.device = device
        self.adam = adam
        self.SDG = SDG
        self.beta1 = beta1
        self.is_l1 = is_l1
        self.is_l2 = is_l2
        self.is_elastic_net = is_elastic_net
        self.is_lr_scheduler = is_lr_scheduler
        self.display = display

    def l1(self, model):
        """
        Calculates the L1 norm (sum of absolute values) of all the parameters in the given model.

        Parameters:
            model (torch.nn.Module): The model whose parameters' L1 norm is to be calculated.

        Returns:
            float: The L1 norm of the model's parameters.

        Raises:
            Exception: If no model is provided.
        """
        if model is not None:
            return (
                torch.norm(input=params, p=1) for params in model.parameters()
            ).sum()
        else:
            raise Exception("Model should be provided".capitalize())

    def l2(self, model):
        """
        Calculates the L2 norm (square root of the sum of squares) of all the parameters in the given model.

        Parameters:
            model (torch.nn.Module): The model whose parameters' L2 norm is to be calculated.

        Returns:
            float: The L2 norm of the model's parameters.

        Raises:
            Exception: If no model is provided.
        """
        if model is not None:
            return (
                torch.norm(input=params, p=2) for params in model.parameters()
            ).sum()
        else:
            raise Exception("Model should be provided".capitalize())

    def elastic_net(self, model):
        """
        Calculates the Elastic Net regularization, which is the sum of L1 and L2 norms, for all the parameters
        in the given model.

        Parameters:
            model (torch.nn.Module): The model whose parameters' Elastic Net regularization is to be calculated.

        Returns:
            float: The sum of L1 and L2 norms of the model's parameters.

        Raises:
            Exception: If no model is provided.
        """
        if model is not None:
            l1 = self.l1(model=model)
            l2 = self.l2(model=model)

            return l1 + l2
        else:
            raise Exception("Model should be provided".capitalize())

    def save_checkpoints(self):
        pass

    def update_discriminator_training(self):
        pass

    def update_generator_training(self):
        pass

    def save_training_images(self):
        pass

    def show_progress(self):
        pass

    def train(self):
        pass

    @staticmethod
    def plot_history():
        pass
