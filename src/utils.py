import os
import joblib
import yaml


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
