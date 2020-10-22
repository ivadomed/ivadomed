import os
import json
from ivadomed.utils import __ivadomed_dir__


def _load_json(config_path):
    with open(config_path, "r") as fhandle:
        default_config = json.load(fhandle)
    return default_config


class ConfigurationManager(object):
    def __init__(self, context_path):
        default_config_path = os.path.join(__ivadomed_dir__, "ivadomed", "config", "config.json")
        self.default_config = _load_json(default_config_path)
        self.context = _load_json(context_path)

    def get_config(self):
        self.default_config.update(self.context)
        return self.default_config
