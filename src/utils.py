import yaml


def params():
    with open("./params.yml", "r") as file:
        return yaml.safe_load(file)
