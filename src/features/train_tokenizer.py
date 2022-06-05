import click
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.processors import TemplateProcessing
from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers.decoders import BPEDecoder


@click.command()
@click.argument('output_tokenizer_path')
@click.argument('input_files_paths', nargs=-1)
def main(output_tokenizer_path, input_files_paths):
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


if __name__ == '__main__':
    main()
