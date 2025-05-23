import toml


def load_config():
    """
    Load the configuration from the pyproject.toml file.
    """
    # Load the pyproject.toml file
    #with open("../config.toml", "r") as f:
    with open("/home/user/faehnrich/ChiRO/src/config(server).toml", "r") as f:
        pyproject = toml.load(f)

    return pyproject
