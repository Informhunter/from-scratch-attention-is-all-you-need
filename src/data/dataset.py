import torch as t
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


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
        source_token_ids = t.LongTensor(self.encoded_sources[index])
        source_attention_mask = t.BoolTensor(self.source_attention_masks[index])
        target_token_ids = t.LongTensor(self.encoded_targets[index])
        target_attention_mask = t.BoolTensor(self.target_attention_masks[index])
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
        return {
            'source_token_ids': pad_sequence(
                [x['source_token_ids'] for x in batch], True, self.tokenizer.token_to_id('[PAD]')
            ),
            'source_attention_masks': pad_sequence(
                [x['source_attention_mask'] for x in batch], True, False
            ),
            'source_texts': [x['source_text'] for x in batch],

            'target_token_ids': pad_sequence(
                [x['target_token_ids'] for x in batch], True, self.tokenizer.token_to_id('[PAD]')
            ),
            'target_attention_masks': pad_sequence(
                [x['target_attention_mask'] for x in batch], True, False
            ),
            'target_texts': [x['target_text'] for x in batch],
        }
