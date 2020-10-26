import json
import os

__ivadomed_dir__ = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


def _load_json(config_path):
    """Load json file content

    Args:
        config_path (str): Path to json file.

    Returns:
        dict: config dictionary.

    """
    with open(config_path, "r") as fhandle:
        default_config = json.load(fhandle)
    return default_config


class ConfigurationManager(object):
    """Configuration file manager

    Args:
        context_path (str): Path to configuration file.

    Attributes:
        context_path (str): Path to configuration file.
        default_config (str): Path to default configuratio file.

    """
    def __init__(self, context_path):
        self.context_path = context_path
        self._validate_path()
        default_config_path = os.path.join(__ivadomed_dir__, "ivadomed", "config", "config_default.json")
        self.default_config = _load_json(default_config_path)
        self.context = _load_json(context_path)

    def get_config(self):
        """Get updated configuration file with all parameters from the default config file.

        Returns:
            dict: Updated configuration dict.
        """
        if self.context['debugging']:
            self._display_differing_keys()
        self.default_config.update(self.context)

        return self.default_config

    def _display_differing_keys(self):
        for key in self.default_config:
            if key not in self.context:
                print(f'Adding the following key in configuration file:\n {key}: {self.default_config[key]}')

    def _validate_path(self):
        """Ensure validity of configuration file path.
        """
        if not os.path.isfile(self.context_path) or not self.context_path.endswith('.json'):
            raise ValueError(
                "\nERROR: The provided configuration file path (.json) is invalid: {}\n".format(self.context_path))
