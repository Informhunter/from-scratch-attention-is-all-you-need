from itertools import chain

import click
import torch as t
import torch.nn.functional as F
import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers
import sacrebleu

from pytorch_lightning.callbacks import ModelCheckpoint

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.processors import TemplateProcessing
from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers import normalizers
from tokenizers.normalizers import NFD, StripAccents
from tokenizers.decoders import BPEDecoder

from src.models.transformer import Transformer
from src.utils.search import beam_search_decode
from src.data.dataset import TranslationDataset, load_data_plaintext, load_data_sgm


def train_tokenizer(training_set_iterator):

    tokenizer = Tokenizer(BPE(unk_token='[UNK]'))
    tokenizer.normalizer = normalizers.Sequence([NFD(), StripAccents()])
    tokenizer.pre_tokenizer = WhitespaceSplit()
    tokenizer.post_processor = TemplateProcessing(
        single='[START] $A [END]',
        special_tokens=[
            ('[START]', 1),
            ('[END]', 2),
        ]
    )
    tokenizer.decoder = BPEDecoder(suffix='</w>')

    trainer = BpeTrainer(
        vocab_size=37000,
        special_tokens=["[UNK]", "[START]", "[END]", "[PAD]"],
        end_of_word_suffix='</w>'
    )

    tokenizer.train_from_iterator(training_set_iterator, trainer=trainer)

    return tokenizer


class TranslatorModel(pl.LightningModule):
    def __init__(self, tokenizer, learning_rate=3e-5):
        super().__init__()
        self.tokenizer = tokenizer
        self.transformer = Transformer(tokenizer.get_vocab_size(), 6, 512, 2048, 8, 64, 64, 0.1)
        self.learning_rate = learning_rate
        self.save_hyperparameters()

    def forward(self, source_token_ids, source_attention_masks, target_token_ids, target_attention_masks):
        return self.transformer(source_token_ids, source_attention_masks, target_token_ids, target_attention_masks)

    def training_step(self, batch, batch_idx):
        preds = self(
            batch['source_token_ids'],
            batch['source_attention_masks'],
            batch['target_token_ids'],
            batch['target_attention_masks']
        )

        loss = self._loss(preds, batch)
        self.log('train_loss', loss.detach().cpu())
        return loss

    def validation_step(self, batch, batch_idx):
        preds = self(
            batch['source_token_ids'],
            batch['source_attention_masks'],
            batch['target_token_ids'],
            batch['target_attention_masks'],
        )

        loss = self._loss(preds, batch)

        encoded_source = self.transformer.encoder_function(batch['source_token_ids'], batch['source_attention_masks'])
        max_decode_length = batch['source_token_ids'].size(0) + 50  # Max length = Input length + 50 (as in the paper)

        decoded_token_ids = beam_search_decode(
            self.transformer,
            encoded_source,
            batch['source_attention_masks'],
            beam_size=4,  # As in the paper
            max_len=max_decode_length,
            alpha=0.6,  # Length penalty as in the paper
        )

        return {
            'val_loss': loss.detach().cpu(),
            'decoded_token_ids': decoded_token_ids,
            'target_texts': batch['target_texts'],
        }

    def validation_epoch_end(self, validation_batches):
        avg_val_loss = t.stack([x['val_loss'] for x in validation_batches]).mean()

        target_texts = list(chain(*[x['target_texts'] for x in validation_batches]))
        decoded_texts = self.tokenizer.decode_batch(
            list(chain(*[x['decoded_token_ids'] for x in validation_batches]))
        )

        val_bleu = sacrebleu.corpus_bleu(decoded_texts, [target_texts])

        self.log('val_bleu', val_bleu.score)
        self.log('val_loss', avg_val_loss)
        self.log('learning_rate', self.learning_rate)

    def _loss(self, preds, batch):
        # preds batch_size x seq_len x vocab_size
        batch_size = preds.shape[0]
        sequence_length = preds.shape[1]

        preds = preds[:, :-1].reshape(batch_size * (sequence_length - 1), -1)
        target = batch['target_token_ids'][:, 1:].reshape(batch_size * (sequence_length - 1))

        loss = F.cross_entropy(preds, target, reduction='none')
        loss = loss.view(batch_size, sequence_length - 1)
        attention_masks = batch['target_attention_masks'][:, 1:]
        loss = t.mean(loss[attention_masks])

        return loss

    def configure_optimizers(self):
        optimizer = t.optim.Adam(self.parameters(), lr=self.learning_rate, eps=1e-09)
        lr_scheduler = t.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.3,
            patience=3,
            min_lr=1e-6,
            verbose=True
        )

        scheduler = {
            'scheduler': lr_scheduler,
            'reduce_on_plateau': True,
            'monitor': 'val_loss',
        }

        return [optimizer], [scheduler]


@click.command()
def main():
    source_datafiles = ['./data/processed/train_en.tsv']
    target_datafiles = ['./data/processed/train_de.tsv']

    source_datafiles_val = ['./data/processed/dev_en.tsv']
    target_datafiles_val = ['./data/processed/dev_de.tsv']

    source_texts = load_data_plaintext(source_datafiles)
    target_texts = load_data_plaintext(target_datafiles)

    source_texts_val = load_data_plaintext(source_datafiles_val)
    target_texts_val = load_data_plaintext(target_datafiles_val)

    try:
        tokenizer = Tokenizer.from_file('./models/tokenizer_en_de.json')
    except Exception:
        tokenizer = train_tokenizer(chain(source_texts, target_texts))
        tokenizer.save('./models/tokenizer_en_de.json')

    train_dataset = TranslationDataset(source_texts, target_texts, tokenizer)
    val_dataset = TranslationDataset(source_texts_val, target_texts_val, tokenizer)

    train_dataloader = t.utils.data.DataLoader(
        train_dataset, shuffle=True, batch_size=16, collate_fn=train_dataset.collate, num_workers=8
    )

    val_dataloader = t.utils.data.DataLoader(
        val_dataset, shuffle=False, batch_size=8, collate_fn=val_dataset.collate, num_workers=8
    )

    model = TranslatorModel(tokenizer, learning_rate=1e-4)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='models/version_0',
        filename='model-{step:05d}-{val_loss:.4f}-{val_bleu:.4f}',
        every_n_val_epochs=1,
        save_top_k=10
    )

    tb_logger = pl_loggers.TensorBoardLogger('./models/logs')

    trainer = pl.Trainer(
        max_epochs=60,
        gpus=1,
        precision=16,
        # accumulate_grad_batches=256,
        val_check_interval=0.05,
        callbacks=[checkpoint_callback]
    )

    trainer.fit(model, train_dataloader, val_dataloader)


if __name__ == '__main__':
    main()
