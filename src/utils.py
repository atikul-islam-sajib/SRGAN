import sys
import os
import joblib
import yaml
import torch
import torch.nn as nn

sys.path.append("src/")

from config import BEST_MODELS, TRAIN_IMAGES


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


def weight_init(m):
    classname = m.__class__.__name__

    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)

    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def device_init(device="mps"):
    if device == "mps":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    elif device == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    else:
        return torch.device("cpu")


def clean_helpers(**kwargs):
    for file in os.listdir(kwargs["path"]):
        os.remove(os.path.join(kwargs["path"], file))


def clean():
    if (os.path.exists(BEST_MODELS)) and (os.path.exists(TRAIN_IMAGES)):
        for path in [BEST_MODELS, TRAIN_IMAGES]:
            clean_helpers(path=path)
    else:
        raise FileExistsError("The directory should be created".capitalize())
