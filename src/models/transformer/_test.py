from typing import (
    List,
    Optional,
)

import click
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from src.features.dataset import (
    IndexedTranslationDataset,
    TranslationDatasetIndex,
    FileIndex,
)
from src.models.transformer.training_module import TranslatorModelTraining
from src.utils.other import (
    parse_devices,
)


@click.command()
@click.option('--regular-checkpoint', 'regular_checkpoint_path', default=None)
@click.option('--averaged-checkpoint', 'averaged_checkpoint_path', default=None)
@click.option('--source', 'source_path', required=True)
@click.option('--target', 'target_path', required=True)
@click.option('--devices', default='0', callback=parse_devices)
def test(
        regular_checkpoint_path: Optional[str],
        averaged_checkpoint_path: Optional[str],
        source_path: Optional[str],
        target_path: Optional[str],
        devices: List[int],
):
    if regular_checkpoint_path:
        model = TranslatorModelTraining.load_from_checkpoint(regular_checkpoint_path)
    elif averaged_checkpoint_path:
        model = torch.load(averaged_checkpoint_path).eval()
    else:
        raise RuntimeError('Need to specify either --regular-checkpoint or --averaged-checkpoint')

    test_dataset_index = TranslationDatasetIndex(
        source_index=FileIndex(filepath=source_path, tokenizer=model.tokenizer),
        target_index=FileIndex(filepath=target_path, tokenizer=model.tokenizer),
        max_length=model.transformer.max_len,
    )

    test_dataset = IndexedTranslationDataset(test_dataset_index)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=8,
        collate_fn=test_dataset.collate,
        num_workers=16,
    )

    trainer = pl.Trainer(
        accelerator='gpu',
        devices=devices,
        strategy='ddp',
    )
    trainer.test(model, dataloaders=test_dataloader)
