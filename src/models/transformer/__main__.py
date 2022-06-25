import os

import click

from src.models.transformer._average_checkpounts import average_checkpoints
from src.models.transformer._inference import inference
from src.models.transformer._test import test
from src.models.transformer._train import train
from src.models.transformer._tune import tune
from src.utils.other import (
    configure_logging,
)


@click.group()
@click.pass_context
def main(ctx: click.Context):
    ctx.ensure_object(dict)
    ctx.obj['main_module_path'] = os.path.abspath(__file__)


if __name__ == '__main__':
    configure_logging()
    main.add_command(train)
    main.add_command(tune)
    main.add_command(average_checkpoints)
    main.add_command(test)
    main.add_command(inference)
    main()
