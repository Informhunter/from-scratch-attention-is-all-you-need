from typing import (
    List,
)

import click
import torch

from src.models.transformer.training_module import TranslatorModelTraining


@click.command()
@click.argument('output_path', type=click.Path())
@click.argument('checkpoint_paths', nargs=-1)
def average_checkpoints(output_path: str, checkpoint_paths: List[str]):
    checkpoints = [
        TranslatorModelTraining.load_from_checkpoint(path)
        for path in checkpoint_paths
    ]
    checkpoint_state_dicts = [
        checkpoint.state_dict()
        for checkpoint in checkpoints
    ]

    new_state_dict = {}
    for key in checkpoint_state_dicts[0].keys():
        new_state_dict[key] = checkpoint_state_dicts[0][key]
        for sd in checkpoint_state_dicts[1:]:
            new_state_dict[key] += sd[key]
        new_state_dict[key] /= len(checkpoint_state_dicts)

    model = checkpoints[0]

    model.load_state_dict(new_state_dict)
    torch.save(model, output_path)
