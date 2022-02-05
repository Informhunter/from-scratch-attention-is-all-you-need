import logging

import click
from tokenizers import Tokenizer

from src.features.dataset import FileIndex


_LOGGER = logging.getLogger(__name__)


@click.command()
@click.argument('input_file_path', type=click.Path(exists=True))
@click.argument('tokenizer_path', type=click.Path(exists=True))
@click.argument('output_file_path', type=click.Path())
def main(input_file_path: str, tokenizer_path: str, output_file_path: str):
    logging.basicConfig(level=logging.INFO)

    tokenizer = Tokenizer.from_file(tokenizer_path)
    index = FileIndex(input_file_path, tokenizer)
    index.to_file(output_file_path)


if __name__ == '__main__':
    main()
