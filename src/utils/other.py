import json
from typing import (
    Any,
    Dict,
    List,
)

import click


def parse_devices(_: click.core.Context, __: click.core.Option, devices: str) -> List[int]:
    return [int(x.strip()) for x in devices.split(',')]


def load_config(_: click.core.Context, __: click.core.Option, config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def set_config_value_by_path(config: Dict[str, Any], path: str, value: Any):
    path_parts = path.split('__')
    last_part = path_parts[-1]
    for path_part in path_parts[:-1]:
        config = config[path_part]
    value_type = type(config[last_part])
    config[last_part] = value_type(value)


def add_unparsed_options_to_config(config: Dict[str, Any], unparsed_args: List[str]):
    args = []
    for arg in unparsed_args:
        for arg_part in arg.split('='):
            args.append(arg_part)
    for i in range(0, len(args), 2):
        key = args[i]
        value = args[i+1]
        if not key.startswith('--config__'):
            raise RuntimeError(f'Wrong option: {key} {value}')
        key = key[10:]  # Remove --config__
        set_config_value_by_path(config, key, value)

