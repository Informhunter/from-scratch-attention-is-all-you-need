from itertools import islice

import torch.utils.data
from torch.utils.data import Dataset, IterableDataset
from torch.nn.utils.rnn import pad_sequence

from tokenizers import Tokenizer

from tqdm import tqdm


def load_data_plaintext(datafiles):
    texts = []
    for f_name in datafiles:
        for line in open(f_name, 'rb'):
            line = line.decode('utf-8')
            texts.append(line.strip())
    return texts


def load_data_sgm(datafiles):
    texts = []
    for f_name in datafiles:
        for line in open(f_name, 'rb'):
            line = line.decode('utf-8')
            if line.startswith('<seg'):
                start_idx = line.find('>') + 1
                end_idx = line.find('</seg>')
                texts.append(line[start_idx:end_idx])
    return texts


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


class TranslationDatasetIndex:
    def __init__(self, source_file_path, target_file_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.source_file_path = source_file_path
        self.target_file_path = target_file_path

        self.source_line_offsets = None
        self.source_line_lengths = None
        self.source_line_token_lengths = None

        self.target_line_offsets = None
        self.target_line_lengths = None
        self.target_line_token_lengths = None

        self.source_file = None
        self.target_file = None

        self._index_samples(source_file_path, target_file_path)

    def _index_samples(self, source_file_path, target_file_path):

        source_offsets = []
        source_lengths = []
        source_token_lengths = []

        target_offsets = []
        target_lengths = []
        target_token_lengths = []

        current_source_offset = 0
        current_target_offset = 0

        source_file = open(source_file_path, 'rb')
        target_file = open(target_file_path, 'rb')
        for source_line, target_line in tqdm(zip(source_file, target_file)):

            source_token_length = self._token_length(source_line)
            target_token_length = self._token_length(target_line)

            if source_token_length < self.max_length and target_token_length < self.max_length:
                source_offsets.append(current_source_offset)
                source_lengths.append(len(source_line))
                source_token_lengths.append(source_token_length)

                target_offsets.append(current_target_offset)
                target_lengths.append(len(target_line))
                target_token_lengths.append(target_token_length)

            current_source_offset += len(source_line)
            current_target_offset += len(target_line)

        self.source_line_offsets = source_offsets
        self.source_line_lengths = source_lengths
        self.source_line_token_lengths = source_token_lengths

        self.target_line_offsets = target_offsets
        self.target_line_lengths = target_lengths
        self.target_line_token_lengths = target_token_lengths

        source_file.close()
        target_file.close()

    def _token_length(self, line):
        line = line.decode('utf-8').strip()
        n_tokens = len(self.tokenizer.encode(line).ids)
        return n_tokens

    def get_item(self, index):
        if self.source_file is None:
            self.source_file = open(self.source_file_path, 'rb')

        if self.target_file is None:
            self.target_file = open(self.target_file_path, 'rb')

        self.source_file.seek(self.source_line_offsets[index])
        source_text = self.source_file.read(self.source_line_lengths[index]).decode('utf-8').strip()

        self.target_file.seek(self.target_line_offsets[index])
        target_text = self.target_file.read(self.target_line_lengths[index]).decode('utf-8').strip()

        encoded_source = self.tokenizer.encode(source_text)
        encoded_target = self.tokenizer.encode(target_text)

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
        return len(self.dataset_index.source_line_offsets)

    def __getitem__(self, index):
        return self.dataset_index.get_item(index)

    def collate(self, batch):
        return generic_collate(batch, self.dataset_index.tokenizer.token_to_id('[PAD]'))


class IndexedPrebatchedTranslationDataset(Dataset):
    def __init__(self, dataset_index: TranslationDatasetIndex, mini_batch_size=1500, maxi_batch_size=100):
        self.dataset_index = dataset_index
        self.mini_batch_size = mini_batch_size
        self.maxi_batch_size = maxi_batch_size
        self.batches = None
        self._prebatch()

    def _prebatch(self):

        batches = []

        it = tqdm(zip(self.dataset_index.source_line_token_lengths, self.dataset_index.target_line_token_lengths))
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
        return generic_collate(batch_items, self.dataset_index.tokenizer.token_to_id('[PAD]'))

    @staticmethod
    def collate(batch):
        return batch[0]


class IterablePrebatchTranslationDataset(IterableDataset):

    def __init__(
            self,
            source_file_path: str,
            target_file_path: str,
            tokenizer: Tokenizer,
            mini_batch_size: int = 5000,
            maxi_batch_size: int = 1000,
            max_length: int = 512,
    ):
        self.source_file_path = source_file_path
        self.target_file_path = target_file_path
        self.tokenizer = tokenizer
        self.num_workers = None
        self.worker_id = None
        self.mini_batch_size = mini_batch_size
        self.maxi_batch_size = maxi_batch_size
        self.max_length = max_length

    def init_worker(self):
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:
            self.num_workers = 1
            self.worker_id = 0
        elif worker_info.num_workers in {0, 1}:
            self.num_workers = 1
            self.worker_id = worker_info.id
        else:
            self.num_workers = worker_info.num_workers
            self.worker_id = worker_info.id

    def _read_n_lines(self, source_file, target_file, n):
        source_texts = []
        target_texts = []

        source_text = source_file.readline().decode('utf-8')
        target_text = target_file.readline().decode('utf-8')
        while source_text and target_text:
            source_text = source_text.strip()
            target_text = target_text.strip()
            source_texts.append(source_text)
            target_texts.append(target_text)
            if len(source_texts) == n:
                break

            for _ in range(self.num_workers - 1):
                source_file.readline()
                target_file.readline()

            source_text = source_file.readline().decode('utf-8')
            target_text = target_file.readline().decode('utf-8')
        return source_texts, target_texts

    def _batch_texts(self, source_texts, target_texts):
        source_encodings = self.tokenizer.encode_batch(source_texts)
        target_encodings = self.tokenizer.encode_batch(target_texts)

        source_target = zip(source_texts, target_texts, source_encodings, target_encodings)

        source_target = (  # Only keep examples that have less than max_length source or target tokens
            (a, b, source_encoding, target_encoding) for a, b, source_encoding, target_encoding in source_target
            if len(source_encoding.ids) <= self.max_length and len(target_encoding.ids) <= self.max_length
        )

        sorted_source_target = sorted(source_target, key=lambda x: len(x[2].ids) + len(x[3].ids))

        batch = []
        batch_size = 0
        for source_text, target_text, source_encoding, target_encoding in sorted_source_target:
            new_item_size = len(source_encoding.ids) + len(target_encoding.ids)
            source_token_ids = torch.LongTensor(source_encoding.ids)
            target_token_ids = torch.LongTensor(target_encoding.ids)

            source_attention_mask = torch.BoolTensor(source_encoding.attention_mask)
            target_attention_mask = torch.BoolTensor(target_encoding.attention_mask)
            item = {
                'source_token_ids': source_token_ids,
                'source_attention_mask': source_attention_mask,
                'target_token_ids': target_token_ids,
                'target_attention_mask': target_attention_mask,
                'source_text': source_text,
                'target_text': target_text,
            }

            if batch_size + new_item_size > self.mini_batch_size:
                yield self.collate(batch)
                batch = []
                batch_size = 0

            batch.append(item)
            batch_size += new_item_size

        if batch:
            yield generic_collate(batch, self.tokenizer.token_to_id('[PAD]'))

    def __iter__(self):

        self.init_worker()

        source_file = open(self.source_file_path, 'rb')
        target_file = open(self.target_file_path, 'rb')

        print('Worker id: ', self.worker_id)

        for i in range(self.worker_id):
            source_file.readline()
            target_file.readline()

        source_texts, target_texts = self._read_n_lines(source_file, target_file, self.maxi_batch_size)
        while source_texts and target_texts:
            for batch in self._batch_texts(source_texts, target_texts):
                yield batch

            source_texts, target_texts = self._read_n_lines(source_file, target_file, self.maxi_batch_size)

        source_file.close()
        target_file.close()

    def __getitem__(self, index):
        raise NotImplementedError()

    @staticmethod
    def collate(batch):
        return batch[0]


class IterableTranslationDataset(IterableDataset):

    def __init__(self, source_file_path, target_file_path, tokenizer, max_length=512):
        self.source_file_path = source_file_path
        self.target_file_path = target_file_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_workers = None
        self.worker_id = None

        self.dataset_length = 0
        with open(source_file_path, 'rb') as f:
            for _ in f:
                self.dataset_length += 1

    def init_worker(self):
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:
            self.num_workers = 1
            self.worker_id = 0
        elif worker_info.num_workers in {0, 1}:
            self.num_workers = 1
            self.worker_id = worker_info.id
        else:
            self.num_workers = worker_info.num_workers
            self.worker_id = worker_info.id

    def __len__(self):
        return self.dataset_length

    def __iter__(self):

        self.init_worker()

        source_file = open(self.source_file_path, 'rb')
        target_file = open(self.target_file_path, 'rb')

        for i in range(self.worker_id):
            source_file.readline()
            target_file.readline()

        source_text = source_file.readline().decode('utf-8')
        target_text = target_file.readline().decode('utf-8')
        while source_text and target_text:
            source_text = source_text.strip()
            target_text = target_text.strip()
            source_encoding = self.tokenizer.encode(source_text)
            target_encoding = self.tokenizer.encode(target_text)

            if len(source_encoding.ids) < self.max_length and len(target_encoding.ids) < self.max_length:

                source_token_ids = torch.LongTensor(source_encoding.ids)
                target_token_ids = torch.LongTensor(target_encoding.ids)

                source_attention_mask = torch.BoolTensor(source_encoding.attention_mask)
                target_attention_mask = torch.BoolTensor(target_encoding.attention_mask)

                yield {
                    'source_token_ids': source_token_ids,
                    'source_attention_mask': source_attention_mask,
                    'target_token_ids': target_token_ids,
                    'target_attention_mask': target_attention_mask,
                    'source_text': source_text,
                    'target_text': target_text,
                }

            for _ in range(self.num_workers - 1):
                source_file.readline()
                target_file.readline()

            source_text = source_file.readline().decode('utf-8')
            target_text = target_file.readline().decode('utf-8')

        source_file.close()
        target_file.close()

    def __getitem__(self, index):
        raise NotImplementedError()

    def collate(self, batch):
        return generic_collate(batch, self.tokenizer.token_to_id('[PAD]'))


class TranslationDataset(Dataset):
    def __init__(self, source_texts, target_texts, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length

        encoded_sources = [x.ids for x in tokenizer.encode_batch(source_texts)]
        encoded_targets = [x.ids for x in tokenizer.encode_batch(target_texts)]

        # Find indices of sequences that are longer than max_length
        too_long_sequence_indices = {
            index for index, source_sequence in enumerate(encoded_sources)
            if len(source_sequence) > self.max_length
        }

        too_long_sequence_indices |= {
            index for index, target_sequence in enumerate(encoded_targets)
            if len(target_sequence) > self.max_length
        }

        # Filter out sequences that are longer than max_length
        encoded_sources = [
            sequence for index, sequence in enumerate(encoded_sources)
            if index not in too_long_sequence_indices
        ]

        encoded_targets = [
            sequence for index, sequence in enumerate(encoded_targets)
            if index not in too_long_sequence_indices
        ]

        source_attention_masks = [[True] * len(x) for x in encoded_sources]
        target_attention_masks = [[True] * len(x) for x in encoded_targets]

        source_texts = [
            text for index, text in enumerate(source_texts)
            if index not in too_long_sequence_indices
        ]

        target_texts = [
            text for index, text in enumerate(target_texts)
            if index not in too_long_sequence_indices
        ]

        self.source_texts = source_texts
        self.target_texts = target_texts
        self.encoded_sources = encoded_sources
        self.source_attention_masks = source_attention_masks
        self.encoded_targets = encoded_targets
        self.target_attention_masks = target_attention_masks

    def __len__(self):
        return len(self.source_texts)

    def __getitem__(self, index):
        source_token_ids = torch.LongTensor(self.encoded_sources[index])
        source_attention_mask = torch.BoolTensor(self.source_attention_masks[index])
        target_token_ids = torch.LongTensor(self.encoded_targets[index])
        target_attention_mask = torch.BoolTensor(self.target_attention_masks[index])
        source_text = self.source_texts[index]
        target_text = self.target_texts[index]

        return {
            'source_token_ids': source_token_ids,
            'source_attention_mask': source_attention_mask,
            'target_token_ids': target_token_ids,
            'target_attention_mask': target_attention_mask,
            'source_text': source_text,
            'target_text': target_text,
        }

    def collate(self, batch):
        return generic_collate(batch, self.tokenizer.token_to_id('[PAD]'))
