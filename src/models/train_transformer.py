import copy
import json
import os.path
import subprocess
import sys
import pickle

from dataclasses import dataclass
from pathlib import Path
from itertools import chain
from typing import Any, Dict, List, Optional

import click
import torch
import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers
import sacrebleu
import optuna

from tokenizers import Tokenizer

from src.models.transformer import Transformer
from src.utils.search import beam_search_decode
from src.utils.train import LRSchedulerVanilla, LRSchedulerNew, Checkpointer
from src.features.dataset import (
    IndexedTranslationDataset,
    IndexedPrebatchedTranslationDataset,
    TranslationDatasetIndex
)


class TranslatorModelTraining(pl.LightningModule):
    def __init__(
            self,
            tokenizer: Tokenizer,
            config: Dict[str, Any],
            source_train_path: Optional[str] = None,
            target_train_path: Optional[str] = None,
            source_val_path: Optional[str] = None,
            target_val_path: Optional[str] = None,
    ):
        super().__init__()

        c = config['model']
        transformer = Transformer(
            tokenizer.get_vocab_size(),
            c['N'],
            c['d_model'],
            c['d_ff'],
            c['h'],
            c['d_k'],
            c['d_v'],
            c['p_drop'],
            c['max_len'],
        )
        self.transformer = transformer
        self.tokenizer = tokenizer
        self.config = config

        if source_train_path is not None:
            train_index_path = os.path.join(
                os.path.dirname(source_train_path),
                'train.index'
            )

            val_index_path = os.path.join(
                os.path.dirname(source_val_path),
                'val.index'
            )

            if not os.path.exists(train_index_path):
                train_dataset_index = TranslationDatasetIndex(
                    source_train_path,
                    target_train_path,
                    tokenizer,
                    max_length=config['dataset']['max_len'],
                )
                with open(train_index_path, 'wb') as f:
                    pickle.dump(train_dataset_index, f)
            else:
                with open(train_index_path, 'rb') as f:
                    train_dataset_index = pickle.load(f)

            if not os.path.exists(val_index_path):
                val_dataset_index = TranslationDatasetIndex(source_val_path, target_val_path, tokenizer)
                with open(val_index_path, 'wb') as f:
                    pickle.dump(val_dataset_index, f)
            else:
                with open(val_index_path, 'rb') as f:
                    val_dataset_index = pickle.load(f)

            self.train_dataset = IndexedPrebatchedTranslationDataset(
                train_dataset_index,
                mini_batch_size=config['dataset']['batch_size'],
                maxi_batch_size=100,
            )
            self.val_dataset = IndexedTranslationDataset(val_dataset_index)

        self.save_hyperparameters(ignore=[
            'source_train_path',
            'target_train_path',
            'source_val_path',
            'target_val_path'
        ])

    def forward(self, source_token_ids, source_attention_masks, target_token_ids, target_attention_masks):
        return self.transformer(source_token_ids, source_attention_masks, target_token_ids, target_attention_masks)

    def training_step(self, batch, batch_idx):
        loss = self.calculate_batch_loss(batch)
        self.log('train_loss', loss.detach().cpu())
        return loss

    def calculate_batch_loss(self, batch):
        preds = self(
            batch['source_token_ids'],
            batch['source_attention_masks'],
            batch['target_token_ids'],
            batch['target_attention_masks']
        )

        loss = self.loss(preds, batch)
        return loss

    def validation_step(self, batch, batch_idx):

        c = self.config['beam_search']

        preds = self(
            batch['source_token_ids'],
            batch['source_attention_masks'],
            batch['target_token_ids'],
            batch['target_attention_masks'],
        )

        loss = self.loss(preds, batch)

        encoded_source = self.transformer.encoder_function(
            batch['source_token_ids'],
            batch['source_attention_masks']
        )

        max_decode_length = (
            batch['source_token_ids'].size(1)
            + c['max_len_factor']
        )  # Max length = Input length + 50 (as in the paper)

        decoded_token_ids = beam_search_decode(
            model=self.transformer,
            encoded_source=encoded_source,
            source_attention_masks=batch['source_attention_masks'],
            beam_size=c['beam_size'],  # As in the paper
            max_len=max_decode_length,
            alpha=c['alpha'],  # Length penalty as in the paper
        )

        return {
            'val_loss': loss.detach().cpu(),
            'decoded_token_ids': decoded_token_ids,
            'target_texts': batch['target_texts'],
        }

    def validation_epoch_end(self, validation_batches):
        avg_val_loss = torch.stack([x['val_loss'] for x in validation_batches]).mean()

        target_texts = list(chain(*[x['target_texts'] for x in validation_batches]))
        decoded_texts = self.tokenizer.decode_batch(
            list(chain(*[x['decoded_token_ids'] for x in validation_batches]))
        )

        val_bleu = sacrebleu.corpus_bleu(decoded_texts, [target_texts])

        self.log('val_bleu', val_bleu.score)
        self.log('val_loss', avg_val_loss)
        self.log('hp_metric', val_bleu.score)

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, validation_batches):
        avg_val_loss = torch.stack([x['val_loss'] for x in validation_batches]).mean()

        target_texts = list(chain(*[x['target_texts'] for x in validation_batches]))
        decoded_texts = self.tokenizer.decode_batch(
            list(chain(*[x['decoded_token_ids'] for x in validation_batches]))
        )

        val_bleu = sacrebleu.corpus_bleu(decoded_texts, [target_texts])

        print('BLEU:', val_bleu.score)
        print('Loss:', avg_val_loss)

    def loss(self, preds, batch):
        """
        :param preds: batch_size x input_seq_len x vocab_size
        :param batch: batch containing target_token_ids and attention_masks
        :return:
        """

        c = self.config['loss']

        batch_size = preds.shape[0]
        sequence_length = preds.shape[1]

        preds = preds[:, :-1].reshape(batch_size * (sequence_length - 1), -1)
        target = batch['target_token_ids'][:, 1:].reshape(batch_size * (sequence_length - 1))

        preds = preds.log_softmax(-1)

        # Label smoothing
        # Positive class gets 1 - label_smoothing
        # Each negative class gets label_smoothing / (vocab_size - 1)

        with torch.no_grad():
            target_dist = torch.zeros_like(preds)
            target_dist.fill_(c['label_smoothing'] / (self.transformer.vocab_size - 1))
            target_dist.scatter_(1, target.unsqueeze(1), 1 - c['label_smoothing'])

        loss = torch.sum(-target_dist * preds, -1)
        attention_masks = batch['target_attention_masks'][:, 1:]
        loss = loss.view(batch_size, sequence_length - 1)
        loss = torch.mean(loss[attention_masks])
        return loss

    def configure_optimizers(self):
        co = self.config['optimizer']
        cs = self.config['scheduler']

        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=co['learning_rate'],
            betas=(co['beta_1'], co['beta_2']),
            eps=co['eps'],
        )

        # scheduler = {
        #     'scheduler': LRSchedulerVanilla(optimizer, self.transformer.d_model, cs['warmup_steps']),
        #     'interval': 'step',
        #     'frequency': 1,
        # }

        scheduler = {
            'scheduler': LRSchedulerNew(optimizer, co['learning_rate'], cs['warmup_steps']),
            'interval': 'step',
            'frequency': 1,
        }

        return [optimizer], [scheduler]

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            collate_fn=self.train_dataset.collate,
            num_workers=16,
            pin_memory=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=8,
            collate_fn=self.val_dataset.collate,
            num_workers=16
        )


def is_zero_rank() -> bool:
    return pl.utilities.distributed._get_rank() == 0


def _init_output_dir(output_dir: str):
    output_dir = Path(output_dir)

    best_checkpoints_dir = output_dir / 'best_checkpoints'
    last_checkpoints_dir = output_dir / 'last_checkpoints'
    logs_dir = output_dir / 'logs'

    if is_zero_rank():
        output_dir.mkdir(parents=True, exist_ok=True)
        best_checkpoints_dir.mkdir(parents=True, exist_ok=False)
        last_checkpoints_dir.mkdir(parents=True, exist_ok=False)

    return best_checkpoints_dir, last_checkpoints_dir, logs_dir


@click.command()
@click.argument('tokenizer_path')
@click.argument('source_train_path')
@click.argument('target_train_path')
@click.argument('source_val_path')
@click.argument('target_val_path')
@click.argument('output_dir')
@click.option('--from-checkpoint', default=None)
@click.option('--config-path', default='config_base.json')
@click.option('--devices', default='0')
@click.option('--no-checkpoints', is_flag=True)
@click.option('--save-metrics', default=None)
def train(
        tokenizer_path: str,
        source_train_path: str,
        target_train_path: str,
        source_val_path: str,
        target_val_path: str,
        output_dir: str,
        from_checkpoint: Optional[str],
        config_path: str,
        devices: str,
        no_checkpoints: bool,
        save_metrics: Optional[str],
):

    pl.seed_everything(42)

    with open(config_path, 'r') as f:
        config = json.load(f)

    best_checkpoints_dir, last_checkpoints_dir, logs_dir = _init_output_dir(output_dir)

    tokenizer = Tokenizer.from_file(tokenizer_path)

    model_training = TranslatorModelTraining(
        tokenizer=tokenizer,
        config=config,
        source_train_path=source_train_path,
        target_train_path=target_train_path,
        source_val_path=source_val_path,
        target_val_path=target_val_path,
    )

    callbacks = []

    lr_monitor = pl.callbacks.LearningRateMonitor('step', False)
    callbacks.append(lr_monitor)

    # early_stopping = pl.callbacks.EarlyStopping(monitor='val_loss', patience=0)
    # callbacks.append(early_stopping)

    if not no_checkpoints:
        best_checkpoints_callback = pl.callbacks.ModelCheckpoint(
            monitor='val_loss',
            dirpath=str(best_checkpoints_dir),
            filename='model-{step:05d}-{val_loss:.4f}-{val_bleu:.4f}',
            every_n_val_epochs=1,
            save_top_k=10
        )
        callbacks.append(best_checkpoints_callback)

        last_n_checkpoint_callback = Checkpointer(
            checkpoint_dir=str(last_checkpoints_dir),
            checkpoint_every_n_batches=10000,
            save_last_k=10,
        )
        callbacks.append(last_n_checkpoint_callback)

    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir=str(logs_dir),
        name=None,
        version=None,
    )

    trainer = pl.Trainer(
        logger=tb_logger,
        max_epochs=config['trainer']['max_epochs'],
        gpus=devices,
        strategy='ddp',
        precision=config['trainer']['precision'],
        accumulate_grad_batches=config['trainer']['accumulate_grad_batches'],
        val_check_interval=0.1,
        callbacks=callbacks,
        resume_from_checkpoint=from_checkpoint,
        num_sanity_val_steps=0,
        enable_checkpointing=not no_checkpoints,
    )

    trainer.fit(model_training)

    if save_metrics is not None and is_zero_rank():
        metrics_path = os.path.join(output_dir, save_metrics)
        with open(metrics_path, 'w') as f:
            metrics = copy.deepcopy(trainer.logged_metrics)
            metrics['train_loss'] = float(metrics['train_loss'])
            json.dump(metrics, f)


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
        ]
        return subprocess.Popen(command)

    def _create_trial_config(self, trial: optuna.Trial, trial_output_dir: Path) -> Path:

        # Sample hyperparameters
        learning_rate = trial.suggest_float('learning_rate', 0.0001, 0.005)
        warmup_steps = trial.suggest_int('warmup_steps', 2000, 6000)

        # Generate config
        with open(self.base_config_path, 'r') as f:
            config = json.load(f)

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
    trainer = pl.Trainer()
    torch.save(model, output_path)


@click.command()
@click.argument('checkpoint_path', type=click.Path(exists=True))
@click.argument('tokenizer_path', type=click.Path(exists=True))
@click.argument('source_test_path', type=click.Path(exists=True))
@click.argument('target_test_path', type=click.Path(exists=True))
def test(checkpoint_path: str, tokenizer_path: str, source_test_path: str, target_test_path: str):
    # model = TranslatorModelTraining.load_from_checkpoint(checkpoint_path)
    model = torch.load(checkpoint_path).eval()
    tokenizer = Tokenizer.from_file(tokenizer_path)

    test_dataset_index = TranslationDatasetIndex(source_test_path, target_test_path, tokenizer)
    test_dataset = IndexedTranslationDataset(test_dataset_index)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=8,
        collate_fn=test_dataset.collate,
        num_workers=0,
    )

    trainer = pl.Trainer(gpus=[0])
    trainer.test(model, test_dataloaders=test_dataloader)


def decode(model: TranslatorModelTraining, text: str):
    encoding = model.tokenizer.encode(text)
    print(encoding.tokens)
    print(encoding.ids)
    source_token_ids = torch.LongTensor(encoding.ids).to(model.device)
    source_attention_mask = torch.BoolTensor(encoding.attention_mask).to(model.device)

    encoded_source = model.transformer.encoder_function(
        source_token_ids.unsqueeze(0),
        source_attention_mask.unsqueeze(0),
    )

    max_decode_length = source_token_ids.size(0) + 50  # Max length = Input length + 50 (as in the paper)

    decoded_token_ids = beam_search_decode(
        model=model.transformer,
        encoded_source=encoded_source,
        source_attention_masks=source_attention_mask.unsqueeze(0),
        beam_size=4,  # As in the paper
        max_len=max_decode_length,
        alpha=0.6,  # Length penalty as in the paper
    )

    print([model.tokenizer.id_to_token(x) for x in decoded_token_ids[0]])
    print(decoded_token_ids[0])

    return model.tokenizer.decode(decoded_token_ids[0])


@click.command()
@click.argument('model_path', type=click.Path(exists=True))
def inference(model_path: str):
    model = TranslatorModelTraining.load_from_checkpoint(model_path).eval()
    model.cuda()

    while True:
        print(decode(model, input('Translate en-de: ')))


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
