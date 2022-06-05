import unittest
import torch
from src.models.transformer import Transformer


class TestTransformer(unittest.TestCase):
    def test_forward_shapes(self):
        with torch.no_grad():
            model = Transformer(37000, 6, 512, 2048, 8, 64, 64, 0.1, 512)
            input_sequence = torch.LongTensor([
                [1, 2, 3, 0, 0],
                [4, 5, 6, 7, 8],
            ])
            input_attention_mask = torch.BoolTensor([
                [True, True, True, False, False],
                [True, True, True, True, True],
            ])

            output_sequence = torch.LongTensor([
                [10, 11, 12],
                [13, 14, 15],
            ])
            output_attention_mask = torch.BoolTensor([
                [True, True, True],
                [True, True, True],
            ])

            result = model(input_sequence, input_attention_mask, output_sequence, output_attention_mask)
            self.assertEqual(result.shape[0], 2)
            self.assertEqual(result.shape[1], 3)
            self.assertEqual(result.shape[2], 37000)


if __name__ == '__main__':
    unittest.main()
