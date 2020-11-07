import json
import os
import collections.abc
from ivadomed import utils as imed_utils


def update(d, u):
    """Update dictionary and nested dictionaries.

    Args:
        d (dict): Source dictionary that is updated by destination dictionary.
        u (dict): Destination dictionary.

    Returns:
        dict: updated dictionary
    """
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def deep_dict_compare(source_dict, dest_dict, keyname=None):
    """Compare and display differences between dictionaries (and nested dictionaries).

    Args:
        source_dict (dict): Source dictionary.
        dest_dict (dict): Destination dictionary.
        keyname (str): Key name to indicate the path to nested parameter.

    """
    for key in dest_dict:
        if key not in source_dict:
            key_str = key if keyname is None else keyname + key
            print(f'    {key_str}: {dest_dict[key]}')

        else:
            if isinstance(dest_dict[key], collections.abc.Mapping):
                deep_dict_compare(source_dict[key], dest_dict[key], key + ": ")


def load_json(config_path):
    """Load json file content

    Args:
        config_path (str): Path to json file.

    Returns:
        dict: config dictionary.

    """
    with open(config_path, "r") as fhandle:
        default_config = json.load(fhandle)
    return default_config


# To ensure retrocompatibility for parameter changes in configuration file
KEY_CHANGE_DICT = {'UNet3D': 'Modified3DUNet'}


class ConfigurationManager(object):
    """Configuration file manager

    Args:
        context_path (str): Path to configuration file.

    Attributes:
        context_path (str): Path to configuration file.
        default_config (dict): Default configuration file.
        context (dict): Provided configuration file.
        updated_config (dict): Update configuration file.

    """
    def __init__(self, context_path):
        self.context_path = context_path
        self.key_change_dict = KEY_CHANGE_DICT
        self._validate_path()
        default_config_path = os.path.join(imed_utils.__ivadomed_dir__, "ivadomed", "config", "config_default.json")
        self.default_config = load_json(default_config_path)
        self.context = load_json(context_path)
        self.updated_config = {}

    def get_config(self):
        """Get updated configuration file with all parameters from the default config file.

        Returns:
            dict: Updated configuration dict.
        """
        self.change_keys()
        self.updated_config = update(self.default_config, self.context)
        if self.updated_config['debugging']:
            self._display_differing_keys()

        return self.updated_config

    def change_keys(self):
        for key in self.key_change_dict:
            if key in self.context:
                self.context[self.key_change_dict[key]] = self.context[key]
                del self.context[key]

    def _display_differing_keys(self):
        """Display differences between dictionaries.
        """
        print('Adding the following keys to the configuration file')
        deep_dict_compare(self.context, self.updated_config)
        print('\n')

    def _validate_path(self):
        """Ensure validity of configuration file path.
        """
        if not os.path.isfile(self.context_path) or not self.context_path.endswith('.json'):
            raise ValueError(
                "\nERROR: The provided configuration file path (.json) is invalid: {}\n".format(self.context_path))
