import pickle
import logging
from itertools import islice
from typing import Tuple

import torch.utils.data
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

import tokenizers
from tokenizers import Tokenizer


_LOGGER = logging.getLogger(__name__)


def generic_collate(batch, pad_token_id):
    return {
        'source_token_ids': pad_sequence(
            sequences=[x['source_token_ids'] for x in batch],
            batch_first=True,
            padding_value=pad_token_id,
        ),
        'source_attention_masks': pad_sequence(
            sequences=[x['source_attention_mask'] for x in batch],
            batch_first=True,
            padding_value=False,
        ),
        'source_texts': [x['source_text'] for x in batch],

        'target_token_ids': pad_sequence(
            sequences=[x['target_token_ids'] for x in batch],
            batch_first=True,
            padding_value=pad_token_id,
        ),
        'target_attention_masks': pad_sequence(
            sequences=[x['target_attention_mask'] for x in batch],
            batch_first=True,
            padding_value=False,
        ),
        'target_texts': [x['target_text'] for x in batch],
    }


class FileIndex:
    def __init__(self, filepath: str, tokenizer: Tokenizer) -> None:
        self.filepath = filepath

        self.line_offsets = None
        self.line_lengths = None
        self.token_counts = None

        self.datafile = None
        self.tokenizer = tokenizer

        self._index_samples()

    def _index_samples(self) -> None:

        _LOGGER.info('Building index for %s', self.filepath)
        offsets = []
        lengths = []
        line_token_counts = []

        current_offset = 0

        with open(self.filepath, 'rb') as f:
            for line in f:
                text = line.decode('utf-8').strip()
                offsets.append(current_offset)
                lengths.append(len(line))
                line_token_counts.append(self._get_token_count(text))
                current_offset += len(line)

        self.line_offsets = offsets
        self.line_lengths = lengths
        self.token_counts = line_token_counts

    def get_text(self, index: int) -> str:
        if self.datafile is None:
            self.datafile = open(self.filepath, 'rb')

        self.datafile.seek(self.line_offsets[index])
        text = self.datafile.read(self.line_lengths[index]).decode('utf-8').strip()

        return text

    def get_encoded_text(self, index: int) -> Tuple[str, tokenizers.Encoding]:
        text = self.get_text(index)
        return text, self.tokenizer.encode(text)

    def _get_token_count(self, text: str):
        return len(self.tokenizer.encode(text).ids)

    @staticmethod
    def from_file(filepath: str):
        with open(filepath, 'rb') as f:
            return pickle.load(f)

    def to_file(self, filepath: str):
        with open(filepath, 'wb') as f:
            return pickle.dump(self, f)


class TranslationDatasetIndex:
    def __init__(self, source_index: FileIndex, target_index: FileIndex, max_length: int):
        self.source_index = source_index
        self.target_index = target_index

        self.max_length = None
        self.indices = None

        self.set_max_length_and_reindex(max_length)

    def set_max_length_and_reindex(self, max_length: int) -> None:
        indices = []

        for i, (source_token_count, target_token_count) in enumerate(zip(self.source_index.token_counts,
                                                                         self.target_index.token_counts)):
            if source_token_count <= max_length and target_token_count <= max_length:
                indices.append(i)
        self.indices = indices
        self.max_length = max_length

    def get_source_token_counts(self):
        return [self.source_index.token_counts[i] for i in self.indices]

    def get_target_token_counts(self):
        return [self.target_index.token_counts[i] for i in self.indices]

    def get_item(self, index: int):
        new_index = self.indices[index]

        source_text, encoded_source = self.source_index.get_encoded_text(new_index)
        target_text, encoded_target = self.target_index.get_encoded_text(new_index)

        source_token_ids = torch.LongTensor(encoded_source.ids)
        source_attention_mask = torch.BoolTensor(encoded_source.attention_mask)

        target_token_ids = torch.LongTensor(encoded_target.ids)
        target_attention_mask = torch.BoolTensor(encoded_target.attention_mask)

        return {
            'source_token_ids': source_token_ids,
            'source_attention_mask': source_attention_mask,
            'target_token_ids': target_token_ids,
            'target_attention_mask': target_attention_mask,
            'source_text': source_text,
            'target_text': target_text,
        }


class IndexedTranslationDataset(Dataset):
    def __init__(self, dataset_index: TranslationDatasetIndex):
        self.dataset_index = dataset_index

    def __len__(self):
        return len(self.dataset_index.indices)

    def __getitem__(self, index: int):
        return self.dataset_index.get_item(index)

    def collate(self, batch):
        return generic_collate(batch, self.dataset_index.source_index.tokenizer.token_to_id('[PAD]'))


class IndexedPrebatchedTranslationDataset(Dataset):
    def __init__(
            self,
            dataset_index: TranslationDatasetIndex,
            mini_batch_size: int = 1500,
            maxi_batch_size: int = 100,
    ):
        self.dataset_index = dataset_index
        self.mini_batch_size = mini_batch_size
        self.maxi_batch_size = maxi_batch_size
        self.batches = None
        self.prebatch()

    def prebatch(self):

        batches = []

        it = zip(self.dataset_index.get_source_token_counts(), self.dataset_index.get_target_token_counts())
        item_index_token_length = (
            (i, a + b)
            for i, (a, b) in enumerate(it)
        )

        maxi_batch = list(islice(item_index_token_length, self.maxi_batch_size))

        while maxi_batch:
            maxi_batch = sorted(maxi_batch, key=lambda x: x[1])

            batch = []
            cumulative_batch_size = 0

            for i, token_length in maxi_batch:

                if cumulative_batch_size + token_length > self.mini_batch_size:
                    batches.append(batch)
                    batch = []
                    cumulative_batch_size = 0

                cumulative_batch_size += token_length
                batch.append(i)

            if len(batch) > 0:
                batches.append(batch)
            maxi_batch = list(islice(item_index_token_length, self.maxi_batch_size))

        self.batches = batches

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, index):
        batch_items = [
            self.dataset_index.get_item(i)
            for i in self.batches[index]
        ]
        return generic_collate(batch_items, self.dataset_index.source_index.tokenizer.token_to_id('[PAD]'))

    @staticmethod
    def collate(batch):
        return batch[0]

