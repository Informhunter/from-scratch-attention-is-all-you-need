import json
import os.path
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import click
import optuna


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
    main_module_path: str

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
            sys.executable, self.main_module_path, 'train',
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
# @click.argument('tokenizer_path', type=click.Path(exists=True))
# @click.argument('source_train_path', type=click.Path(exists=True))
# @click.argument('target_train_path', type=click.Path(exists=True))
# @click.argument('source_val_path', type=click.Path(exists=True))
# @click.argument('target_val_path', type=click.Path(exists=True))
# @click.argument('base_config_path', type=click.Path(exists=True))
# @click.argument('study_output_dir', type=click.Path())
@click.option('--devices', default='0')
@click.option('--max-epochs', default=1)
@click.option('--n-trials', default=100)
@click.option('--study-name', default='transformer-tuning')
@click.pass_context
def tune(
        ctx: click.Context,
        # tokenizer_path: str,
        # source_train_path: str,
        # target_train_path: str,
        # source_val_path: str,
        # target_val_path: str,
        # base_config_path: str,
        # study_output_dir: str,
        devices: str,
        max_epochs: int,
        n_trials: int,
        study_name: str,
):

    print(ctx.obj)
    return

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
        main_module_path=ctx.obj['main_module_path'],
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
