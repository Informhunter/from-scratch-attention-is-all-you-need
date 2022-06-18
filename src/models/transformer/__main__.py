import copy
import json
import os.path
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
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
import optuna
import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers
import torch
from pytorch_lightning.strategies.ddp import DDPStrategy
from tokenizers import Tokenizer
from torch.utils.data import DataLoader

from src.features.dataset import (
    IndexedTranslationDataset,
    TranslationDatasetIndex,
    FileIndex,
)
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
@click.argument('tokenizer_path')
@click.argument('train_source_index_path')
@click.argument('train_target_index_path')
@click.argument('val_source_index_path')
@click.argument('val_target_index_path')
@click.argument('output_dir')
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


@dataclass
class OptunaObjective:
    tokenizer_path: str
    source_train_path: str
    target_train_path: str
    source_val_path: str
    target_val_path: str
    base_config_path: str
    study_output_dir: str
    devices: str
    max_epochs: int

    def __call__(self, trial: optuna.Trial) -> float:
        study_output_dir = Path(self.study_output_dir)

        trial_output_dir = study_output_dir / f'trial_{trial.number}'
        trial_output_dir.mkdir(exist_ok=False)

        trial_config_path = self._create_trial_config(trial, trial_output_dir)

        popen = self._launch_trial(trial_output_dir, trial_config_path)
        popen.wait()

        metric = self._read_metric(trial_output_dir)

        return metric

    def _launch_trial(self, trial_output_dir: Path, trial_config_path: Path) -> subprocess.Popen:
        command = [
            sys.executable, os.path.abspath(__file__), 'train',
            self.tokenizer_path, self.source_train_path, self.target_train_path,
            self.source_val_path, self.target_val_path, str(trial_output_dir),
            '--config-path', str(trial_config_path),
            '--devices', self.devices,
            '--no-checkpoints',
            '--save-metrics', 'metrics.json',
            '--early-stopping',
        ]
        return subprocess.Popen(command)

    def _create_trial_config(self, trial: optuna.Trial, trial_output_dir: Path) -> Path:

        # Sample hyperparameters
        learning_rate = trial.suggest_float('learning_rate', 0.0001, 0.005)
        warmup_steps = trial.suggest_int('warmup_steps', 2000, 6000)

        # Generate config
        with open(self.base_config_path, 'r') as f:
            config = json.load(f)

        config['scheduler']['type'] = 'new'
        config['optimizer']['learning_rate'] = learning_rate
        config['scheduler']['warmup_steps'] = warmup_steps
        config['trainer']['max_epochs'] = self.max_epochs

        trial_config_path = trial_output_dir / 'config.json'

        with trial_config_path.open('w') as f:
            json.dump(config, f)

        return trial_config_path

    @staticmethod
    def _read_metric(trial_output_dir: Path) -> float:
        metrics_path = trial_output_dir / 'metrics.json'
        with metrics_path.open('r') as f:
            metrics = json.load(f)
        return metrics['val_bleu']


@click.command()
@click.argument('tokenizer_path', type=click.Path(exists=True))
@click.argument('source_train_path', type=click.Path(exists=True))
@click.argument('target_train_path', type=click.Path(exists=True))
@click.argument('source_val_path', type=click.Path(exists=True))
@click.argument('target_val_path', type=click.Path(exists=True))
@click.argument('base_config_path', type=click.Path(exists=True))
@click.argument('study_output_dir', type=click.Path())
@click.option('--devices', default='0')
@click.option('--max-epochs', default=1)
@click.option('--n-trials', default=100)
@click.option('--study-name', default='transformer-tuning')
def tune(
    tokenizer_path: str,
    source_train_path: str,
    target_train_path: str,
    source_val_path: str,
    target_val_path: str,
    base_config_path: str,
    study_output_dir: str,
    devices: str,
    max_epochs: int,
    n_trials: int,
    study_name: str,
):

    objective = OptunaObjective(
        tokenizer_path=tokenizer_path,
        source_train_path=source_train_path,
        target_train_path=target_train_path,
        source_val_path=source_val_path,
        target_val_path=target_val_path,
        devices=devices,
        base_config_path=base_config_path,
        study_output_dir=study_output_dir,
        max_epochs=max_epochs,
    )

    study_output_dir = Path(study_output_dir)
    study_output_dir.mkdir(exist_ok=True)

    study_storage = os.path.join('sqlite:///', study_output_dir, 'study.db')
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(
        direction='maximize',
        sampler=sampler,
        storage=study_storage,
        study_name=study_name,
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=n_trials)


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


@click.command()
@click.option('--regular-checkpoint', 'regular_checkpoint_path', default=None)
@click.option('--averaged-checkpoint', 'averaged_checkpoint_path', default=None)
@click.option('--source-index', 'source_index_path')
@click.option('--target-index', 'target_index_path')
@click.option('--devices', default='0', callback=parse_devices)
def test(
        regular_checkpoint_path: Optional[str],
        averaged_checkpoint_path: Optional[str],
        source_index_path: Optional[str],
        target_index_path: Optional[str],
        devices: List[int],
):
    if regular_checkpoint_path:
        model = TranslatorModelTraining.load_from_checkpoint(regular_checkpoint_path)
    elif averaged_checkpoint_path:
        model = torch.load(averaged_checkpoint_path).eval()
    else:
        raise RuntimeError('Need to specify either --regular-checkpoint or --averaged-checkpoint')

    if not source_index_path or not target_index_path:
        raise RuntimeError('Need to specify --source-index and --target-index')

    test_dataset_index = TranslationDatasetIndex(
        source_index=FileIndex.from_file(source_index_path),
        target_index=FileIndex.from_file(target_index_path),
        max_length=model.transformer.max_len,
    )

    test_dataset = IndexedTranslationDataset(test_dataset_index)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=8,
        collate_fn=test_dataset.collate,
        num_workers=16,
    )

    trainer = pl.Trainer(accelerator='gpu', devices=devices)
    trainer.test(model, dataloaders=test_dataloader)


@click.command()
@click.argument('model_path', type=click.Path(exists=True))
def inference(model_path: str):
    model = TranslatorModelTraining.load_from_checkpoint(model_path).eval()
    model.cuda()

    while True:
        text = input('Translate en-de: ')
        encoding = model.tokenizer.encode(text)
        print(encoding.tokens)
        print(encoding.ids)
        source_token_ids = torch.LongTensor(encoding.ids).to(model.device)
        source_attention_mask = torch.BoolTensor(encoding.attention_mask).to(model.device)

        decoded_token_ids = model.decode(
            source_token_ids=source_token_ids.unsqueeze(0),
            source_attention_masks=source_attention_mask.unsqueeze(0),
        )

        print([model.tokenizer.id_to_token(x) for x in decoded_token_ids])
        print(decoded_token_ids)
        print(model.tokenizer.decode(decoded_token_ids))


@click.group()
def main():
    pass


if __name__ == '__main__':
    main.add_command(train)
    main.add_command(tune)
    main.add_command(average_checkpoints)
    main.add_command(test)
    main.add_command(inference)
    main()
