#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import List, Optional, Union
import yacs.config

from advtex_init_align.utils.logging import logger


class Config(yacs.config.CfgNode):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, new_allowed=True)


CN = Config

DEFAULT_CONFIG_DIR = "configs/"
CONFIG_FILE_SEPARATOR = ","

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = CN()
_C.SEED = 100


def get_config(
    config_paths: Optional[Union[List[str], str]] = None,
    opts: Optional[list] = None,
) -> CN:
    r"""Create a unified config with default values overwritten by values from
    :p:`config_paths` and overwritten by options from :p:`opts`.

    :param config_paths: List of config paths or string that contains comma
        separated list of config paths.
    :param opts: Config options (keys, values) in a list (e.g., passed from
        command line into the config. For example,
        :py:`opts = ['FOO.BAR', 0.5]`. Argument can be used for parameter
        sweeping or quick tests.
    """
    config = _C.clone()
    if config_paths:
        if isinstance(config_paths, str):
            if CONFIG_FILE_SEPARATOR in config_paths:
                config_paths = config_paths.split(CONFIG_FILE_SEPARATOR)
            else:
                config_paths = [config_paths]

        for config_path in config_paths:
            config.merge_from_file(config_path)

    if opts:
        config.merge_from_list(opts)

    config.freeze()
    return config


def update_config_log(config: Config, run_type: str, log_dir: str):
    config.defrost()
    config.LOG_DIR = log_dir
    config.LOG_FILE = os.path.join(log_dir, f"{run_type}.log")
    config.INFO_DIR = os.path.join(log_dir, f"infos")
    config.CHECKPOINT_FOLDER = os.path.join(log_dir, "checkpoints")
    config.TENSORBOARD_DIR = os.path.join(log_dir, "tb")
    config.VIS_DIR = os.path.join(log_dir, "vis")
    config.freeze()

    os.makedirs(config.LOG_DIR, exist_ok=True)
    os.makedirs(config.INFO_DIR, exist_ok=True)
    os.makedirs(config.CHECKPOINT_FOLDER, exist_ok=True)
    os.makedirs(config.TENSORBOARD_DIR, exist_ok=True)
    os.makedirs(config.VIS_DIR, exist_ok=True)
    return config


def _assert_with_logging(cond, msg):
    if not cond:
        logger.debug(msg)
    assert cond, msg


def _valid_type(value, allow_cfg_node=False):
    return (type(value) in _VALID_TYPES) or (
        allow_cfg_node and isinstance(value, yacs.config.CfgNode)
    )


# CfgNodes can only contain a limited set of valid types
_VALID_TYPES = {tuple, list, str, int, float, bool, type(None)}


def convert_cfg_to_dict(cfg_node, key_list=[]):
    if not isinstance(cfg_node, yacs.config.CfgNode):
        _assert_with_logging(
            _valid_type(cfg_node),
            "Key {} with value {} is not a valid type; valid types: {}".format(
                ".".join(key_list), type(cfg_node), _VALID_TYPES
            ),
        )
        return cfg_node
    else:
        cfg_dict = dict(cfg_node)
        for k, v in cfg_dict.items():
            cfg_dict[k] = convert_cfg_to_dict(v, key_list + [k])
        return cfg_dict
