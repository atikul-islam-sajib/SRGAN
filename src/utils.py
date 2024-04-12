import os
import joblib
import yaml
import torch
import torch.nn as nn


def dump(value, filename):
    if (value is not None) and (filename is not None):
        joblib.dump(value=value, filename=filename)
    else:
        raise Exception("Value or filename not provided")


def load(filename):
    if filename is not None:
        return joblib.load(filename=filename)
    else:
        raise Exception("Filename not provided")


def params():
    with open("./params.yml", "r") as file:
        return yaml.safe_load(file)


def weight_int(m):
    classname = m.__class__.__name__

    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)

    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
