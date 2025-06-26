import os
import yaml
import logging

from typing import Any

log = logging.getLogger(__name__)

def load_config_file(config_file: str,
                     dict_to_update: dict[str, Any],
                     key_to_check: str,
                     ) -> dict[str, Any]:
    """Load configuration from a YAML file.

    Args:
        config_file (str, optional): Path to the configuration file. If not provided, it will look for a default config file.
        dict_to_update (dict[str, Any], optional): Dictionary to update with parameters from the config file. Defaults to None.
        key_to_check (str, optional): Key to check in the config file. If this key is not found, an error will be raised. Defaults to None.

    Returns:
        dict[str, Any]: The updated dictionary with parameters from the config file.
    """
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        # Update parameters from the config file
        if key_to_check in config:
            custom_config = config.get(key_to_check, {})
            print(f"Custom config for {key_to_check}: {custom_config}")
            if all(k in custom_config for k in dict_to_update.keys()):
                dict_to_update.update(custom_config)
                log.info(f"Loaded {key_to_check} parameters from config file")
            else:
                log.error(f"Incomplete {key_to_check} parameters in config file")
                exit("Please check the config file for missing parameters.")
        else:
            log.error(f"{key_to_check} not found in config file")
            exit(f"Please check the config file for the {key_to_check} section.")
    else:
        log.error(f"Config file {config_file} does not exist.")
        exit(f"Please provide a valid config file at {config_file}.")
        
    return dict_to_update