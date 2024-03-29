from itertools import chain
from typing import (
    Any,
    Dict,
    List,
    Optional,
)

import pytorch_lightning as pl
import sacrebleu
import torch
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
from tokenizers import Tokenizer
from torch.utils.data import DataLoader

from src.features.dataset import (
    FileIndex,
    TranslationDatasetIndex,
    IndexedPrebatchedTranslationDataset,
    IndexedTranslationDataset,
)
from src.models.transformer.model import Transformer
from src.utils.search import beam_search_decode
from src.utils.train import (
    LRSchedulerNew,
    LRSchedulerVanilla,
)


class TranslatorModelTraining(pl.LightningModule):
    def __init__(
            self,
            tokenizer: Tokenizer,
            config: Dict[str, Any],
            train_source_index_path: Optional[str] = None,
            train_target_index_path: Optional[str] = None,
            val_source_index_path: Optional[str] = None,
            val_target_index_path: Optional[str] = None,
    ):
        super().__init__()

        c = config['model']
        transformer = Transformer(
            vocab_size=tokenizer.get_vocab_size(),
            n_layers=c['n_layers'],
            d_model=c['d_model'],
            d_ff=c['d_ff'],
            h=c['h'],
            d_k=c['d_k'],
            d_v=c['d_v'],
            p_drop=c['p_drop'],
            max_len=c['max_len'],
            checkpoint_gradients=c['checkpoint_gradients'],
        )
        self.transformer = transformer
        self.tokenizer = tokenizer
        self.config = config
        self.train_source_index_path = train_source_index_path
        self.train_target_index_path = train_target_index_path
        self.val_source_index_path = val_source_index_path
        self.val_target_index_path = val_target_index_path

        self._load_dataset_indexes()

        self.save_hyperparameters(ignore=[
            'source_train_path',
            'target_train_path',
            'source_val_path',
            'target_val_path',
        ])

    def _load_dataset_indexes(self) -> None:

        if self.train_source_index_path is not None and self.train_target_index_path is not None:
            train_index = TranslationDatasetIndex(
                source_index=FileIndex.from_file(self.train_source_index_path),
                target_index=FileIndex.from_file(self.train_target_index_path),
                max_length=self.config['dataset']['max_len'],
            )
            self.train_dataset = IndexedPrebatchedTranslationDataset(
                dataset_index=train_index,
                mini_batch_size=self.config['dataset']['batch_size'],
                maxi_batch_size=self.config['dataset']['maxi_batch_size'],
                max_batch_length=self.config['dataset']['max_batch_length'],
            )

        if self.val_source_index_path is not None and self.val_target_index_path is not None:
            val_index = TranslationDatasetIndex(
                FileIndex.from_file(self.val_source_index_path),
                FileIndex.from_file(self.val_target_index_path),
                max_length=self.config['model']['max_len'],
            )
            self.val_dataset = IndexedTranslationDataset(val_index)

    def forward(
            self,
            source_token_ids: torch.LongTensor,
            source_attention_masks: torch.BoolTensor,
            target_token_ids: torch.LongTensor,
            target_attention_masks: torch.BoolTensor,
    ) -> torch.FloatTensor:
        return self.transformer(
            input_sequence=source_token_ids,
            input_attention_mask=source_attention_masks,
            output_sequence=target_token_ids,
            output_attention_mask=target_attention_masks,
        )

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.FloatTensor:
        preds = self(
            batch['source_token_ids'],
            batch['source_attention_masks'],
            batch['target_token_ids'],
            batch['target_attention_masks'],
        )

        loss = self.loss(preds, batch)

        batch_length = batch['source_token_ids'].shape[0]
        sequence_length = (
                batch['source_token_ids'].shape[1] +
                batch['source_token_ids'].shape[2]
        )

        self.log('train_loss', loss.detach().cpu().item())
        self.log('batch_length', batch_length)
        self.log('sequence_length', sequence_length)
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, Any]:

        preds = self(
            batch['source_token_ids'],
            batch['source_attention_masks'],
            batch['target_token_ids'],
            batch['target_attention_masks'],
        )

        loss = self.loss(preds, batch)

        decoded_token_ids = self.decode(batch['source_token_ids'], batch['source_attention_masks'])

        return {
            'val_loss': loss.detach().cpu(),
            'decoded_token_ids': decoded_token_ids,
            'target_texts': batch['target_texts'],
        }

    def validation_epoch_end(self, validation_batches: List[Dict[str, Any]]) -> None:
        avg_val_loss = torch.stack([x['val_loss'] for x in validation_batches]).mean()

        target_texts = list(chain(*[x['target_texts'] for x in validation_batches]))
        decoded_texts = self.tokenizer.decode_batch(
            list(chain(*[x['decoded_token_ids'] for x in validation_batches]))
        )

        val_bleu = sacrebleu.corpus_bleu(decoded_texts, [target_texts])

        self.log('val_bleu', val_bleu.score)
        self.log('val_loss', avg_val_loss)
        self.log('hp_metric', val_bleu.score)

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, Any]:
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, validation_batches: List[Dict[str, Any]]) -> None:
        test_avg_loss = torch.stack([x['val_loss'] for x in validation_batches]).mean()

        target_texts = list(chain(*[x['target_texts'] for x in validation_batches]))
        decoded_texts = self.tokenizer.decode_batch(
            list(chain(*[x['decoded_token_ids'] for x in validation_batches]))
        )

        test_bleu = sacrebleu.corpus_bleu(decoded_texts, [target_texts])
        self.log_dict({
            'test_bleu': test_bleu.score,
            'test_avg_loss':  test_avg_loss,
        })

    def loss(self, preds: torch.FloatTensor, batch: Dict[str, Any]) -> torch.FloatTensor:
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

        if cs['type'] == 'vanilla':
            scheduler = {
                'scheduler': LRSchedulerVanilla(optimizer, self.transformer.d_model, cs['warmup_steps']),
                'interval': 'step',
                'frequency': 1,
            }
        elif cs['type'] == 'new':
            scheduler = {
                'scheduler': LRSchedulerNew(optimizer, co['learning_rate'], cs['warmup_steps']),
                'interval': 'step',
                'frequency': 1,
            }
        else:
            raise RuntimeError(f'Wrong config scheduler type "{cs["type"]}"')

        return [optimizer], [scheduler]

    def decode(self, source_token_ids: torch.LongTensor, source_attention_masks: torch.BoolTensor) -> List[List[int]]:

        c = self.config['beam_search']
        encoded_source = self.transformer.encoder_function(
            source_token_ids,
            source_attention_masks,
        )

        max_decode_length = (
                source_token_ids.size(1)
                + c['max_len_factor']
        )  # Max length = Input length + 50 (as in the paper)

        decoded_token_ids = beam_search_decode(
            model=self.transformer,
            encoded_source=encoded_source,
            source_attention_masks=source_attention_masks,
            beam_size=c['beam_size'],  # As in the paper
            max_len=max_decode_length,
            alpha=c['alpha'],  # Length penalty as in the paper
        )

        return decoded_token_ids

    def train_dataloader(self) -> DataLoader:
        c = self.config['train_dataloader']
        return DataLoader(
            self.train_dataset,
            collate_fn=self.train_dataset.collate,
            num_workers=c['num_workers'],
            pin_memory=True,
            prefetch_factor=c['prefetch_factor'],
        )

    def val_dataloader(self) -> DataLoader:
        c = self.config['val_dataloader']
        return DataLoader(
            self.val_dataset,
            batch_size=c['batch_size'],
            collate_fn=self.val_dataset.collate,
            num_workers=c['num_workers'],
            prefetch_factor=c['prefetch_factor'],
        )

    # Overwriting with default code because otherwise get MisconfigurationException for custom schedulers
    def lr_scheduler_step(self, scheduler: Any, optimizer_idx: int, metric: Optional[Any]) -> None:
        if metric is None:
            scheduler.step()
        else:
            scheduler.step(metric)
