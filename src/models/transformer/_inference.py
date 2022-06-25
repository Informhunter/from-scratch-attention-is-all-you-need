import click
import torch

from src.models.transformer.training_module import TranslatorModelTraining


@click.command()
@click.option('--model', 'model_path', type=click.Path(exists=True), required=True)
@click.option('--device', default='cpu')
def inference(model_path: str, device: str):
    device = torch.device(device)

    model = TranslatorModelTraining.load_from_checkpoint(model_path).eval()
    model.to(device)

    while True:
        text = input('Translate en-de: ')
        encoding = model.tokenizer.encode(text)
        print(encoding.tokens)
        print(encoding.ids)
        source_token_ids = torch.LongTensor(encoding.ids).to(model.device)
        source_attention_mask = torch.BoolTensor(encoding.attention_mask).to(model.device)

        decoded_token_ids = model.decode(
            source_token_ids=source_token_ids.unsqueeze(0),
            source_attention_masks=source_attention_mask.unsqueeze(0),
        )[0]

        print(decoded_token_ids)
        print([model.tokenizer.id_to_token(x) for x in decoded_token_ids])
        print(model.tokenizer.decode(decoded_token_ids))
