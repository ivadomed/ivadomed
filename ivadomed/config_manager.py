import json
import collections.abc
from loguru import logger
from pathlib import Path
from ivadomed import utils as imed_utils
from ivadomed.keywords import ConfigKW, LoaderParamsKW, SplitDatasetKW, DataTestingKW
import copy


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
            # If source dictionary has keys that the destination dict doesn't have, keep these keys
            if k in d and isinstance(d[k], collections.abc.Mapping) and not isinstance(v, collections.abc.Mapping):
                pass
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
            logger.info(f'    {key_str}: {dest_dict[key]}')

        else:
            if isinstance(dest_dict[key], collections.abc.Mapping):
                if isinstance(source_dict[key], collections.abc.Mapping):
                    deep_dict_compare(source_dict[key], dest_dict[key], key + ": ")
                # In case a new dictionary appears in updated file
                else:
                    deep_dict_compare(source_dict, dest_dict[key], key + ": ")


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
KEY_CHANGE_DICT = {'UNet3D': ConfigKW.MODIFIED_3D_UNET, 'bids_path': LoaderParamsKW.PATH_DATA,
                   'log_directory': ConfigKW.PATH_OUTPUT}
KEY_SPLIT_DATASET_CHANGE_LST = ['method', 'center_test']


class ConfigurationManager(object):
    """Configuration file manager.

    Args:
        path_context (str): Path to configuration file.
    Attributes:
        path_context (str): Path to configuration file.
        config_default (dict): Default configuration file from ``ivadomed`` package.
        context_original (dict): Provided configuration file.
        config_updated (dict): Updated configuration file.
    """
    def __init__(self, path_context):
        self.path_context = path_context
        self.key_change_dict = KEY_CHANGE_DICT
        self.key_split_dataset_change_lst = KEY_SPLIT_DATASET_CHANGE_LST
        self._validate_path()
        default_config_path = str(Path(imed_utils.__ivadomed_dir__, "ivadomed", "config", "config_default.json"))
        self.config_default = load_json(default_config_path)
        self.context_original = load_json(path_context)
        self.config_updated = {}

    @property
    def config_updated(self):
        return self._config_updated

    @config_updated.setter
    def config_updated(self, config_updated):
        if config_updated == {}:
            context = copy.deepcopy(self.context_original)
            self.change_keys(context, list(context.keys()))
            config_updated = update(self.config_default, context)
            self.change_keys_values(config_updated[ConfigKW.SPLIT_DATASET], config_updated[ConfigKW.SPLIT_DATASET].keys())

        self._config_updated = config_updated
        if config_updated['debugging']:
            self._display_differing_keys()

    def get_config(self):
        """Get updated configuration file with all parameters from the default config file.
        Returns:
            dict: Updated configuration dict.
        """
        return self.config_updated

    def change_keys(self, context, keys):
        for k in keys:
            # Verify if key is still in the dict
            if k in context:
                if k == "NumpyToTensor":
                    del context[k]
                    continue
                v = context[k]
                # Verify if value is a dictionary
                if isinstance(v, collections.abc.Mapping):
                    self.change_keys(v, list(v.keys()))
                else:
                    # Change keys from the key_change_dict
                    for key in self.key_change_dict:
                        if key in context:
                            context[self.key_change_dict[key]] = context[key]
                            del context[key]

    def change_keys_values(self, config_updated, keys):
        for k in self.key_split_dataset_change_lst:
            if k in keys:
                value = config_updated[k]
                if k == 'method' and value == "per_center":
                    config_updated[SplitDatasetKW.DATA_TESTING][DataTestingKW.DATA_TYPE] = "institution_id"
                if k == 'center_test' and \
                        config_updated[SplitDatasetKW.DATA_TESTING][DataTestingKW.DATA_TYPE] == "institution_id" and \
                value is not None:
                    config_updated[SplitDatasetKW.DATA_TESTING][DataTestingKW.DATA_VALUE] = value
                del config_updated[k]

    def _display_differing_keys(self):
        """Display differences between dictionaries.
        """
        logger.info('Adding the following keys to the configuration file')
        deep_dict_compare(self.context_original, self.config_updated)
        logger.info('\n')

    def _validate_path(self):
        """Ensure validity of configuration file path.
        """
        if not Path(self.path_context).exists():
            raise ValueError(
                f"\nERROR: The provided configuration file path (.json) does not exist: "
                f"{Path(self.path_context).absolute()}\n")
        elif Path(self.path_context).is_dir():
            raise IsADirectoryError(f"ERROR: The provided configuration file path (.json) is a directory not a file: "
                                    f"{Path(self.path_context).absolute()}\n")
        elif not Path(self.path_context).is_file():
            raise FileNotFoundError(f"ERROR: The provided configuration file path (.json) is not found: "
                                    f"{Path(self.path_context).absolute()}\n")
        elif self.path_context.endswith('.yaml') or self.path_context.endswith('.yml'):
            raise ValueError(
                f"\nERROR: The provided configuration file path (.json) is a yaml file not a json file, "
                f"yaml files are not yet supported: {Path(self.path_context).absolute()}\n")
        elif not self.path_context.endswith('.json'):
            raise ValueError(
                f"\nERROR: The provided configuration file path (.json) is not a json file: "
                f"{Path(self.path_context).absolute()}\n")
