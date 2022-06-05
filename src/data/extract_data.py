import logging
import tarfile
from dataclasses import dataclass
from typing import List

import click

_LOGGER = logging.getLogger(__name__)


@dataclass
class _DataFile:
    tar_path: str
    datafile_path: str
    format: str
    skip_empty_lines: bool = False

    def iterate_lines(self):
        with tarfile.open(self.tar_path, 'r:gz') as tar_file:
            with tar_file.extractfile(self.datafile_path) as f:
                lines = f.read()

        lines = lines.decode('utf-8').split('\n')
        for line in lines:
            if self.format == 'sgm':
                if line.startswith('<seg'):
                    start_idx = line.find('>') + 1
                    end_idx = line.find('</seg>')
                    line = line[start_idx:end_idx]
                else:
                    continue
            yield line.strip()


_TRAIN_EN_FILES = [
    _DataFile(tar_path='data/raw/training-parallel-commoncrawl.tgz',
              datafile_path='commoncrawl.de-en.en',
              format='text'),
    _DataFile(tar_path='data/raw/training-parallel-europarl-v7.tgz',
              datafile_path='training/europarl-v7.de-en.en',
              format='text'),
    _DataFile(tar_path='data/raw/training-parallel-nc-v9.tgz',
              datafile_path='training/news-commentary-v9.de-en.en',
              format='text')
]

_TRAIN_DE_FILES = [
    _DataFile(tar_path='data/raw/training-parallel-commoncrawl.tgz',
              datafile_path='commoncrawl.de-en.de',
              format='text'),
    _DataFile(tar_path='data/raw/training-parallel-europarl-v7.tgz',
              datafile_path='training/europarl-v7.de-en.de',
              format='text'),
    _DataFile(tar_path='data/raw/training-parallel-nc-v9.tgz',
              datafile_path='training/news-commentary-v9.de-en.de',
              format='text')
]

_DEV_EN_FILES = [
    _DataFile(tar_path='data/raw/dev.tgz',
              datafile_path='dev/newstest2013.en',
              format='text',
              skip_empty_lines=True),
]

_DEV_DE_FILES = [
    _DataFile(tar_path='data/raw/dev.tgz',
              datafile_path='dev/newstest2013.de',
              format='text',
              skip_empty_lines=True),
]

_TEST_FILTERED_EN_FILES = [
    _DataFile(tar_path='data/raw/test-filtered.tgz',
              datafile_path='test/newstest2014-deen-src.en.sgm',
              format='sgm'),
]
_TEST_FILTERED_DE_FILES = [
    _DataFile(tar_path='data/raw/test-filtered.tgz',
              datafile_path='test/newstest2014-deen-ref.de.sgm',
              format='sgm'),
]

_TEST_FULL_EN_FILES = [
    _DataFile(tar_path='data/raw/test-filtered.tgz',
              datafile_path='test/newstest2014-deen-src.en.sgm',
              format='sgm'),
]

_TEST_FULL_DE_FILES = [
    _DataFile(tar_path='data/raw/test-filtered.tgz',
              datafile_path='test/newstest2014-deen-ref.de.sgm',
              format='sgm'),
]


def _datafile_callback(_, __, input_file_infos: List[List[str]]):
    return [
        _DataFile(
            tar_path=input_file_info[0],
            datafile_path=input_file_info[1],
            format=input_file_info[3],
        )
        for input_file_info in input_file_infos
    ]


def _extract(input_files: List[_DataFile], output_file_path: str):
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        for input_file in input_files:
            _LOGGER.info('Extracting %s:%s into %s', input_file.tar_path, input_file.datafile_path, output_file_path)
            for line in input_file.iterate_lines():
                if not input_file.skip_empty_lines or line:
                    output_file.write(line + '\n')


@click.command()
@click.option(
    '--input-file',
    'input_files',
    nargs=3,
    multiple=True,
    callback=_datafile_callback,
    help='Tar file name, filename of datafile (inside tar file), datafile format (text/sgm).'
)
@click.option('--output-file', 'output_file_path', help='Output file name to extract data to.')
def extract(input_files: List[_DataFile], output_file_path: str):
    _extract(input_files=input_files, output_file_path=output_file_path)


@click.command()
def default_extract():
    _extract(input_files=_TRAIN_EN_FILES, output_file_path='data/processed/train.en')
    _extract(input_files=_TRAIN_DE_FILES, output_file_path='data/processed/train.de')

    _extract(input_files=_DEV_EN_FILES, output_file_path='data/processed/dev.en')
    _extract(input_files=_DEV_DE_FILES, output_file_path='data/processed/dev.de')

    _extract(input_files=_TEST_FULL_EN_FILES, output_file_path='data/processed/test_full.en')
    _extract(input_files=_TEST_FULL_DE_FILES, output_file_path='data/processed/test_full.de')

    _extract(input_files=_TEST_FILTERED_EN_FILES, output_file_path='data/processed/test_filtered.en')
    _extract(input_files=_TEST_FILTERED_DE_FILES, output_file_path='data/processed/test_filtered.de')


@click.group()
def main():
    pass


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main.add_command(extract)
    main.add_command(default_extract)
    main()
