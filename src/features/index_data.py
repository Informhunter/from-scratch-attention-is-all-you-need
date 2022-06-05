import logging
from typing import List

import click
from tokenizers import Tokenizer

from src.features.dataset import FileIndex
from src.utils.logging import configure_logging

_LOGGER = logging.getLogger(__name__)


@click.command()
@click.argument('input_file_paths', nargs=-1)
@click.argument('tokenizer_path', nargs=1)
def main(input_file_paths: List[str], tokenizer_path: str):
    configure_logging()

    tokenizer = Tokenizer.from_file(tokenizer_path)
    for input_file_path in input_file_paths:
        index = FileIndex(input_file_path, tokenizer)
        index.to_file(input_file_path + '.index')


if __name__ == '__main__':
    main()
