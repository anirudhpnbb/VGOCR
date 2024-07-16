# utils.py

import json

def load_config(config_path):
    """
    Load the configuration from a JSON file.

    Args:
    - config_path (str): Path to the JSON configuration file.

    Returns:
    - config (dict): Configuration dictionary.
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config
