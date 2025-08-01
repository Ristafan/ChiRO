import toml
import os
import json
from pathlib import Path
from datetime import datetime

from src.Training.TrainingParams import TrainingParams


def load_path_config():
    # Load the pyproject.toml file
    with open("../config.toml", "r") as f:
    #with open("/home/user/faehnrich/ChiRO/src/config_rolf.toml", "r") as f:
    #with open("/cluster/raid/home/f60047174/ChiRO/src/config_gamarello.toml", "r") as f:
        pyproject = toml.load(f)

    return pyproject


def create_experiment_dir(config, base_dir):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_path = Path(base_dir) / f"run_{timestamp}"
    exp_path.mkdir(parents=True, exist_ok=False)

    training_params_dict = config.__dict__

    # Save config
    with open(exp_path / "config.json", "w") as f:
        json.dump(training_params_dict, f, indent=2)

    return exp_path


def update_experiment_configs(exp_path, config):
    training_params_dict = config.__dict__

    # Save config
    with open(exp_path / "config.json", "w") as f:
        json.dump(training_params_dict, f, indent=2)

    # Save the path to the experiment directory in the config
    config.experiment_path = str(exp_path)

    return config


def log_metrics(exp_path, epoch, metrics):
    log_path = exp_path / "metrics.json"
    if log_path.exists():
        with open(log_path, "r") as f:
            logs = json.load(f)
    else:
        logs = {}

    logs[str(epoch)] = metrics

    with open(log_path, "w") as f:
        json.dump(logs, f, indent=2)
