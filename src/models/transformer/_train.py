import copy
import json
import os.path
from pprint import pprint
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
)

import click
import fsspec.utils
import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers
from pytorch_lightning.strategies.ddp import DDPStrategy
from tokenizers import Tokenizer

from src.models.transformer.training_module import TranslatorModelTraining
from src.utils.other import (
    parse_devices,
    add_unparsed_options_to_config,
    load_config,
)
from src.utils.train import Checkpointer


def _init_output_dir(output_dir: str) -> Tuple[str, str, str]:
    best_checkpoints_dir = os.path.join(output_dir, 'best_checkpoints')
    last_checkpoints_dir = os.path.join(output_dir, 'last_checkpoints')
    logs_dir = os.path.join(output_dir, 'logs')

    @pl.utilities.rank_zero.rank_zero_only
    def _zero_rank_makedirs(*args, **kwargs) -> None:
        os.makedirs(*args, **kwargs)

    if fsspec.utils.get_protocol(output_dir) == 'file':
        _zero_rank_makedirs(output_dir, exist_ok=True)
        _zero_rank_makedirs(best_checkpoints_dir, exist_ok=True)
        _zero_rank_makedirs(last_checkpoints_dir, exist_ok=True)

    return best_checkpoints_dir, last_checkpoints_dir, logs_dir


@click.command(context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
@click.option('--tokenizer', 'tokenizer_path', required=True)
@click.option('--train-source-index', 'train_source_index_path', required=True)
@click.option('--train-target-index', 'train_target_index_path', required=True)
@click.option('--val-source-index', 'val_source_index_path', required=True)
@click.option('--val-target-index', 'val_target_index_path', required=True)
@click.option('--output-dir', 'output_dir', required=True)
@click.option('--from-checkpoint', default=None)
@click.option('--config', default='model_configs/config_base.json', callback=load_config)
@click.option('--devices', default='0', callback=parse_devices)
@click.option('--no-checkpoints', is_flag=True)
@click.option('--save-metrics', 'save_metrics_path', default=None)
@click.option('--early-stopping', is_flag=True)
@click.option('--seed', default=42)
@click.option('--project-name', default='AIAYN')
@click.pass_context
def train(
        ctx: click.core.Context,
        tokenizer_path: str,
        train_source_index_path: str,
        train_target_index_path: str,
        val_source_index_path: str,
        val_target_index_path: str,
        output_dir: str,
        from_checkpoint: Optional[str],
        config: Dict[str, Any],
        devices: List[int],
        no_checkpoints: bool,
        save_metrics_path: Optional[str],
        early_stopping: bool,
        seed: int,
        project_name: str,
):

    pl.seed_everything(seed)

    add_unparsed_options_to_config(config, ctx.args)

    pprint(config)

    best_checkpoints_dir, last_checkpoints_dir, logs_dir = _init_output_dir(output_dir)

    tokenizer = Tokenizer.from_file(tokenizer_path)

    model_training = TranslatorModelTraining(
        tokenizer=tokenizer,
        config=config,
        train_source_index_path=train_source_index_path,
        train_target_index_path=train_target_index_path,
        val_source_index_path=val_source_index_path,
        val_target_index_path=val_target_index_path,
    )

    callbacks = []

    lr_monitor = pl.callbacks.LearningRateMonitor(
        logging_interval='step',
        log_momentum=False,
    )
    callbacks.append(lr_monitor)

    if early_stopping:
        early_stopping_callback = pl.callbacks.EarlyStopping(monitor='val_loss', patience=3)
        callbacks.append(early_stopping_callback)

    if not no_checkpoints:
        best_checkpoints_callback = pl.callbacks.ModelCheckpoint(
            monitor='val_loss',
            dirpath=best_checkpoints_dir,
            filename='model-{step:05d}-{val_loss:.4f}-{val_bleu:.4f}',
            every_n_epochs=1,
            save_top_k=config['trainer']['save_best_k_checkpoints'],
        )
        callbacks.append(best_checkpoints_callback)

        last_n_checkpoint_callback = Checkpointer(
            checkpoint_dir=last_checkpoints_dir,
            checkpoint_every_n_steps=10000,
            save_last_k=config['trainer']['save_last_k_checkpoints'],
        )
        callbacks.append(last_n_checkpoint_callback)

    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir=logs_dir,
        name=None,
        version=None,
    )

    wandb_logger = pl_loggers.WandbLogger(project=project_name)

    trainer = pl.Trainer(
        logger=[tb_logger, wandb_logger],
        gradient_clip_val=config['trainer']['gradient_clip_val'],
        max_epochs=config['trainer']['max_epochs'],
        accelerator='auto',
        devices=devices,
        strategy=DDPStrategy(find_unused_parameters=False),
        precision=config['trainer']['precision'],
        accumulate_grad_batches=config['trainer']['accumulate_grad_batches'],
        val_check_interval=config['trainer']['val_check_interval'],
        callbacks=callbacks,
        resume_from_checkpoint=from_checkpoint,
        num_sanity_val_steps=config['trainer']['num_sanity_val_checks'],
        enable_checkpointing=not no_checkpoints,
    )

    trainer.fit(model_training)

    @pl.utilities.rank_zero_only
    def _save_metrics():
        if save_metrics_path is not None:
            metrics_path = os.path.join(output_dir, save_metrics_path)
            with fsspec.open(metrics_path, 'w') as f:
                metrics = copy.deepcopy(trainer.logged_metrics)
                metrics['train_loss'] = float(metrics['train_loss'])
                json.dump(metrics, f)
    _save_metrics()
