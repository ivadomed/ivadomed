import json
import os

from ivadomed.utils import __ivadomed_dir__


def _load_json(config_path):
    with open(config_path, "r") as fhandle:
        default_config = json.load(fhandle)
    return default_config


class ConfigurationManager(object):
    def __init__(self, context_path):
        self.context_path = context_path
        self._validate_path()
        default_config_path = os.path.join(__ivadomed_dir__, "ivadomed", "config", "config_default.json")
        self.default_config = _load_json(default_config_path)
        self.context = _load_json(context_path)

    def get_config(self):
        self.default_config.update(self.context)
        return self.default_config

    def _validate_path(self):
        if not os.path.isfile(self.context_path) or not self.context_path.endswith('.json'):
            raise ValueError(
                "\nERROR: The provided configuration file path (.json) is invalid: {}\n".format(self.context_path))
