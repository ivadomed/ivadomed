#!/usr/bin/env python
"""
Launcher for training with the possibility to specify hyperparameters. This script is to be used by wandb-sweep.
https://docs.wandb.ai/guides/sweeps

Input: template JSON ivadomed config file, which will be edited by this script to insert the random parameters.

Resumption of interrupted runs with
    python train_wandb_sweep_hyperparam.py --resume

"""

import sys

import wandb
from wandb.keras import WandbCallback


defaults = dict(
    dropout=0.2,
    hidden_layer_size=128,
    layer_1_size=16,
    layer_2_size=32,
    learn_rate=0.01,
    decay=1e-6,
    momentum=0.9,
    epochs=27,
    )

resume = sys.argv[-1] == "--resume"
wandb.init(config=defaults, resume=resume)

# This is where the parameters are read from the YAML file
config = wandb.config

# Build ivadomed's config file based on the generated parameters
# TODO

# Call to ivadomed
ivadomed.main()
