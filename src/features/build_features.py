import click
import pandas as pd
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.processors import TemplateProcessing
from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers.decoders import BPEDecoder


@click.command()
@click.argument('output_tokenizer_path')
@click.argument('input_files_paths', nargs=-1)
def train_tokenizer(output_tokenizer_path, input_files_paths):
    tokenizer = Tokenizer(BPE(unk_token='[UNK]'))
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

    tokenizer.train(input_files_paths, trainer=trainer)
    tokenizer.save(output_tokenizer_path)


@click.command()
@click.argument('tokenizer_path')
@click.argument('input_file_path')
@click.argument('output_file_path')
def tokenize(tokenizer_path, input_file_path, output_file_path):
    tokenizer = Tokenizer.from_file(tokenizer_path)
    texts = []
    for line in open(input_file_path, 'rb'):
        line = line.decode('utf-8')
        texts.append(line.strip())

    encoding = tokenizer.encode_batch(texts)

    df = pd.DataFrame()
    df['text'] = texts
    df['token_ids'] = [','.join(str(y) for y in x.ids) for x in encoding]
    df.to_csv(output_file_path, sep='\t', index=None)


@click.group()
def main():
    pass


if __name__ == '__main__':
    main.add_command(train_tokenizer)
    main.add_command(tokenize)
    main()
