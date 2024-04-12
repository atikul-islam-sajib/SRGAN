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
        pass

    def l2(self, model):
        pass

    def elastic_net(self, model):
        pass

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
