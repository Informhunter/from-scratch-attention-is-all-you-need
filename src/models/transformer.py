import math
import torch
from torch import nn


class Transformer(nn.Module):

    def __init__(self, vocab_size, N, d_model, d_ff, h, d_k, d_v, p_drop):
        super().__init__()

        self.vocab_size = vocab_size
        self.N = N
        self.d_model = d_model
        self.d_ff = d_ff
        self.h = h
        self.d_k = d_k
        self.d_v = d_v
        self.p_drop = p_drop

        self.encoder = TransformerEncoder(N, d_model, d_ff, h, d_k, d_v, p_drop)
        self.decoder = TransformerDecoder(N, d_model, d_ff, h, d_k, d_v, p_drop)
        self.dropout = nn.Dropout(p=p_drop)

        self.positional_encoding = PositionalEncoding(d_model, 1024)

        self.input_embedding = nn.Embedding(vocab_size, d_model)
        self.output_embedding = nn.Embedding(vocab_size, d_model)
        self.next_token_classifier = nn.Linear(d_model, vocab_size, bias=False)
        self.output_embedding.weight = self.input_embedding.weight
        self.next_token_classifier.weight = self.input_embedding.weight

    def forward(self, input_sequence, input_attention_mask, output_sequence, output_attention_mask):
        """
        :param input_sequence: batch_size x input_seq_len
        :param input_attention_mask: batch_size x input_seq_len
        :param output_sequence: batch_size x output_seq_len
        :param output_attention_mask: batch_size x output_seq_len
        :return: next_token_logits: batch_size x output_seq_len x vocab_size
        """

        import pdb; pdb.set_trace()
        encoded_input = self.encoder_function(input_sequence, input_attention_mask)
        next_token_logits = self.decoder_function(
            encoded_input, input_attention_mask,
            output_sequence, output_attention_mask
        )

        return next_token_logits

    def encoder_function(self, input_sequence, input_attention_mask):
        encoded_input_sequence = self.input_embedding(input_sequence) * math.sqrt(self.d_model)
        encoded_input_sequence = self.positional_encoding(encoded_input_sequence)
        encoded_input_sequence = self.dropout(encoded_input_sequence)
        encoded_input_sequence = self.encoder(encoded_input_sequence, input_attention_mask)
        return encoded_input_sequence

    def decoder_function(self, encoded_input_sequence, input_attention_mask, output_sequence, output_attention_mask):
        encoded_output_sequence = self.output_embedding(output_sequence) * math.sqrt(self.d_model)
        encoded_output_sequence = self.positional_encoding(encoded_output_sequence)
        encoded_output_sequence = self.dropout(encoded_output_sequence)
        encoded_output_sequence = self.decoder(
            encoded_input_sequence, input_attention_mask,
            encoded_output_sequence, output_attention_mask
        )
        next_token_logits = self.next_token_classifier(encoded_output_sequence)
        return next_token_logits


class TransformerEncoder(nn.Module):
    def __init__(self, N, d_model, d_ff, h, d_k, d_v, p_drop):
        super().__init__()

        self.N = N
        self.d_model = d_model
        self.d_ff = d_ff
        self.h = h
        self.d_k = d_k
        self.d_v = d_v
        self.p_drop = p_drop

        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, d_ff, h, d_k, d_v, p_drop)
            for i in range(N)
        ])

    def forward(self, encoded_input_sequence, input_attention_mask):
        """
        :param encoded_input_sequence: batch_size x input_seq_len x d_model
        :param input_attention_mask: batch_size x input_seq_len
        :return: batch_size x input_seq_len x d_model
        """

        for encoder_layer in self.encoder_layers:
            encoded_input_sequence = encoder_layer(encoded_input_sequence, input_attention_mask)

        return encoded_input_sequence


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, d_ff, h, d_k, d_v, p_drop):
        super().__init__()

        self.d_model = d_model
        self.d_ff = d_ff
        self.h = h
        self.d_k = d_k
        self.d_v = d_v
        self.p_drop = p_drop

        self.multihead_attention = MultiHeadAttention(d_model, h, d_k, d_v, False)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.layer_norm_1 = nn.LayerNorm(d_model)
        self.layer_norm_2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=p_drop)

    def forward(self, encoded_input, attention_mask):
        """
        :param encoded_input: batch_size x input_seq_len x d_model
        :param attention_mask: batch_size x input_seq_len
        :return: batch_size x input_seq_len x d_model
        """
        encoded_input = self.layer_norm_1(
            encoded_input + self.multihead_attention(
                encoded_input,
                encoded_input,
                encoded_input,
                attention_mask,
                attention_mask,
            )
        )
        encoded_input = self.layer_norm_2(
            encoded_input + self.feed_forward(encoded_input)
        )
        return encoded_input


class TransformerDecoder(nn.Module):
    def __init__(self, N, d_model, d_ff, h, d_k, d_v, p_drop):
        super().__init__()

        self.N = N
        self.d_model = d_model
        self.d_ff = d_ff
        self.h = h
        self.d_k = d_k
        self.d_v = d_v
        self.p_drop = p_drop

        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, d_ff, h, d_k, d_v, p_drop)
            for i in range(N)
        ])

    def forward(self, input_sequence_encoding, input_attention_mask, output_sequence_encoding, output_attention_mask):
        for decoder_layer in self.decoder_layers:
            output_sequence_encoding = decoder_layer(
                input_sequence_encoding, input_attention_mask, output_sequence_encoding, output_attention_mask
            )
        return output_sequence_encoding


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, h, d_k, d_v, p_drop):
        super().__init__()

        self.d_model = d_model
        self.d_ff = d_ff
        self.h = h
        self.d_k = d_k
        self.d_v = d_v
        self.p_drop = p_drop

        self.multihead_attention_1 = MultiHeadAttention(d_model, h, d_k, d_v, True)
        self.multihead_attention_2 = MultiHeadAttention(d_model, h, d_k, d_v, False)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.layer_norm_1 = nn.LayerNorm(d_model)
        self.layer_norm_2 = nn.LayerNorm(d_model)
        self.layer_norm_3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=p_drop)

    def forward(self, input_sequence_encoding, input_attention_mask, output_sequence_encoding, output_attention_mask):

        # Apply self-attention
        output_sequence_encoding_delta = self.multihead_attention_1(
            output_sequence_encoding,
            output_sequence_encoding,
            output_sequence_encoding,
            output_attention_mask,
            output_attention_mask,
        )

        output_sequence_encoding_delta = self.dropout(output_sequence_encoding_delta)
        output_sequence_encoding = self.layer_norm_1(output_sequence_encoding + output_sequence_encoding_delta)

        # Apply attention over encoder output
        output_sequence_encoding_delta = self.multihead_attention_2(
            output_sequence_encoding,
            input_sequence_encoding,
            input_sequence_encoding,
            output_attention_mask,
            input_attention_mask,
        )
        output_sequence_encoding_delta = self.dropout(output_sequence_encoding_delta)
        output_sequence_encoding = self.layer_norm_2(output_sequence_encoding + output_sequence_encoding_delta)

        # Apply feed-forward layers
        output_sequence_encoding_delta = self.feed_forward(output_sequence_encoding)
        output_sequence_encoding_delta = self.dropout(output_sequence_encoding_delta)
        output_sequence_encoding = self.layer_norm_3(output_sequence_encoding + output_sequence_encoding_delta)

        return output_sequence_encoding


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, h, d_k, d_v, mask_back_connections):
        super().__init__()
        self.h = h
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.mask_back_connections = mask_back_connections

        self.queries_transform_w = nn.Parameter(torch.empty(h, 1, d_model, d_k))
        # self.queries_transform_b = nn.Parameter(t.empty(h, 1, 1, d_k))
        self.keys_transform_w = nn.Parameter(torch.empty(h, 1, d_model, d_k))
        # self.keys_transform_b = nn.Parameter(t.empty(h, 1, 1, d_k))
        self.values_transform_w = nn.Parameter(torch.empty(h, 1, d_model, d_v))
        # self.values_transform_b = nn.Parameter(t.empty(h, 1, 1, d_v))

        nn.init.xavier_uniform_(self.queries_transform_w)
        # t.nn.init.xavier_uniform_(self.queries_transform_b)
        nn.init.xavier_uniform_(self.keys_transform_w)
        # t.nn.init.xavier_uniform_(self.keys_transform_b)
        nn.init.xavier_uniform_(self.values_transform_w)
        # t.nn.init.xavier_uniform_(self.values_transform_b)

        self.output_transform = nn.Parameter(torch.empty(h * d_v, d_model))
        nn.init.xavier_uniform_(self.output_transform)

    def forward(self, queries, keys, values, queries_attention_mask, keys_attention_mask):
        """
        :param queries: batch_size x queries_len x d_model
        :param keys: batch_size x keys_len x d_model
        :param values: batch_size x keys_len x d_model
        :param queries_attention_mask: batch_size x queries_len
        :param keys_attention_mask: batch_size x keys_len
        :return: batch_size x seq_len x d_model
        """

        batch_size = queries.shape[0]
        queries_len = queries.shape[1]

        # Project queries, keys, and values

        # matmul(
        #   batch_size x seq_len x d_model,
        #   h x 1 x seq_len x d_model x (d_k or d_v)
        # ) -> h x batch_size x seq_len x (d_k or d_v)

        queries = torch.matmul(queries, self.queries_transform_w)
        keys = torch.matmul(keys, self.keys_transform_w)
        values = torch.matmul(values, self.values_transform_w)

        # Calculate attention weights

        attention_weights = torch.matmul(
            queries,
            torch.transpose(keys, -1, -2)
        ) / math.sqrt(self.d_k)

        # Mask attention
        self._mask_attention(attention_weights, queries_attention_mask, keys_attention_mask)

        attention_weights = torch.softmax(attention_weights, -1)

        # Calculate attention outputs
        attention_head_outputs = torch.matmul(attention_weights, values)
        attention_head_outputs = attention_head_outputs.permute(1, 2, 0, 3)

        # Concatenate attention outputs and project back to d_model
        attention_head_outputs = attention_head_outputs.reshape(batch_size, queries_len, -1)
        result = torch.matmul(attention_head_outputs, self.output_transform)

        return result

    def _mask_attention(self, attention_weights, queries_attention_mask, keys_attention_mask):
        """
        For all False values in attention_mask set corresponding values in attention to -inf.
        If need to mask backwards connections set all attention values below diagonal to -inf.

        :param attention_weights: attention weights (before softmax) to update:
            h x batch_size x queries_len x keys_len
        :param queries_attention_mask: boolean tensor indicating non-PAD tokens in queries: batch_size x queries_len
        :param keys_attention_mask: boolean tensor indicating non-PAD tokens in keys: batch_size x keys_len
        :return: None
        """

        h = attention_weights.shape[0]
        batch_size = attention_weights.shape[1]
        queries_len = attention_weights.shape[2]
        keys_len = attention_weights.shape[3]
        device = queries_attention_mask.device

        set_to_minus_inf = torch.full(
            (1, batch_size, queries_len, keys_len),
            False,
            dtype=torch.bool,
            device=device,
        )

        queries_attention_mask = ~(
            queries_attention_mask
            .view(batch_size, queries_len, 1)
            .expand(batch_size, queries_len, keys_len)
        )

        keys_attention_mask = ~(
            keys_attention_mask
            .view(batch_size, 1, keys_len)
            .expand(batch_size, queries_len, keys_len)
        )

        set_to_minus_inf = torch.logical_or(
            set_to_minus_inf,
            queries_attention_mask
        )

        set_to_minus_inf = torch.logical_or(
            set_to_minus_inf,
            keys_attention_mask
        )

        if self.mask_back_connections:
            indices = torch.triu_indices(queries_len, keys_len, 1)
            set_to_minus_inf[:, :, indices[0], indices[1]] = True

        attention_weights[
            set_to_minus_inf.expand(h, batch_size, queries_len, keys_len)
        ] = -math.inf


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.linear_1(x)
        x = torch.relu(x)
        x = self.linear_2(x)
        return x


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len):
        super().__init__()
        self.positional_encodings = nn.Parameter(
            torch.FloatTensor(max_len, d_model),
            requires_grad=False
        )

        for pos in range(max_len):
            for i in range(d_model):
                if i % 2 == 0:
                    self.positional_encodings[pos][i] = math.sin(pos / 10000 ** (2 * i / d_model))
                else:
                    self.positional_encodings[pos][i] = math.cos(pos / 10000 ** (2 * i / d_model))

    def forward(self, encoded_sequence):
        batch_size = encoded_sequence.shape[0]
        sequence_length = encoded_sequence.shape[1]
        positional_encoding = self.positional_encodings[:sequence_length, :]
        positional_encoding = positional_encoding.unsqueeze(0).repeat(batch_size, 1, 1)
        return encoded_sequence + positional_encoding
