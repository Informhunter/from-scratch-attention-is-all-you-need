from typing import Any, Dict, List, Optional

import pytorch_lightning as pl
from tokenizers import Tokenizer

from src.models.transformer import Transformer


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
