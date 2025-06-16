import toml


def load_config():
    """
    Load the configuration from the pyproject.toml file.
    """
    # Load the pyproject.toml file
    #with open("../config.toml", "r") as f:
    #with open("/home/user/faehnrich/ChiRO/src/config_rolf.toml", "r") as f:
    with open("/cluster/raid/home/f60047174/ChiRO/src/config_gamarello.toml", "r") as f:
        pyproject = toml.load(f)

    return pyproject
