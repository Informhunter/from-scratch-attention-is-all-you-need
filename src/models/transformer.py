import math
import torch
from torch import nn


class Transformer(nn.Module):

    def __init__(
            self,
            vocab_size: int,
            N: int,
            d_model: int,
            d_ff: int,
            h: int,
            d_k: int,
            d_v: int,
            p_drop: float,
            max_len: int,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.N = N
        self.d_model = d_model
        self.d_ff = d_ff
        self.h = h
        self.d_k = d_k
        self.d_v = d_v
        self.p_drop = p_drop
        self.max_len = max_len

        self.encoder = TransformerEncoder(N, d_model, d_ff, h, d_k, d_v, p_drop)
        self.decoder = TransformerDecoder(N, d_model, d_ff, h, d_k, d_v, p_drop)
        self.dropout = nn.Dropout(p=p_drop)

        self.positional_encoding = PositionalEncoding(d_model, max_len)

        # Input embedding, output embedding and next_token_classifier share same weights
        self.input_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
        )

        output_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
        )

        next_token_classifier = nn.Linear(
            in_features=d_model,
            out_features=vocab_size,
            bias=False,
        )
        output_embedding.weight = self.input_embedding.weight
        next_token_classifier.weight = self.input_embedding.weight

        self.output_embedding = output_embedding
        self.next_token_classifier = next_token_classifier

    def forward(
            self,
            input_sequence: torch.LongTensor,
            input_attention_mask: torch.BoolTensor,
            output_sequence: torch.LongTensor,
            output_attention_mask: torch.BoolTensor,
    ) -> torch.FloatTensor:
        """
        :param input_sequence: batch_size x input_seq_len - token ids for input sequences
        :param input_attention_mask: batch_size x input_seq_len - attention masks for input sequences
        (False for padding tokens, True otherwise)
        :param output_sequence: batch_size x output_seq_len - token_ids for output sequence
        :param output_attention_mask: batch_size x output_seq_len - attention masks for input sequences
        (False for padding tokens, True otherwise)
        :return next_token_logits: batch_size x output_seq_len x vocab_size - next token probability distribution
        logits for each position in output_sequence. Apply softmax to acquire probability distributions.

        """
        encoded_input_sequence = self.encoder_function(input_sequence, input_attention_mask)
        next_token_logits = self.decoder_function(
            encoded_input_sequence, input_attention_mask,
            output_sequence, output_attention_mask
        )

        return next_token_logits

    def encoder_function(self, input_sequence: torch.LongTensor, input_attention_mask: torch.BoolTensor):
        """
        Encode input sequences into vector representations. Each position in input sequences gets a corresponding
        vector of size d_model.
        :param input_sequence: batch_size x input_seq_len - token ids for input sequences
        :param input_attention_mask: batch_size x input_seq_len - attention masks for input sequences
        (False for padding tokens, True otherwise)
        :return encoded_input_sequence: batch_size x input_seq_len x d_model
        """
        encoded_input_sequence = self.input_embedding(input_sequence)
        encoded_input_sequence = self.positional_encoding(encoded_input_sequence)
        encoded_input_sequence = self.dropout(encoded_input_sequence)
        encoded_input_sequence = self.encoder(encoded_input_sequence, input_attention_mask)
        return encoded_input_sequence

    def decoder_function(
            self,
            encoded_input_sequence: torch.LongTensor,
            input_attention_mask: torch.BoolTensor,
            output_sequence: torch.LongTensor,
            output_attention_mask: torch.BoolTensor,
    ) -> torch.FloatTensor:
        """
        Based on encoded input sequences and output sequences predict next tokens for each position in input sequences.
        :param encoded_input_sequence: batch_size x input_seq_len x d_model
        :param input_attention_mask: batch_size x input_seq_len
        :param output_sequence: batch_size x output_seq_len
        :param output_attention_mask: batch_size x output_seq_len
        :return next_token_logits: batch_size x output_seq_len x vocab_size
        """
        encoded_output_sequence = self.output_embedding(output_sequence)
        encoded_output_sequence = self.positional_encoding(encoded_output_sequence)
        encoded_output_sequence = self.dropout(encoded_output_sequence)
        encoded_output_sequence = self.decoder(
            input_sequence_encoding=encoded_input_sequence,
            input_attention_mask=input_attention_mask,
            output_sequence_encoding=encoded_output_sequence,
            output_attention_mask=output_attention_mask,
        )
        # Scale logits by d_model ** -0.5
        next_token_logits = self.next_token_classifier(encoded_output_sequence) / math.sqrt(self.d_model)
        return next_token_logits


class TransformerEncoder(nn.Module):
    def __init__(
            self,
            N: int,
            d_model: int,
            d_ff: int,
            h: int,
            d_k: int,
            d_v: int,
            p_drop: float,
    ):
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
            for _ in range(N)
        ])

    def forward(self, encoded_input_sequence, input_attention_mask):
        """
        :param encoded_input_sequence: batch_size x input_seq_len x d_model
        :param input_attention_mask: batch_size x input_seq_len
        :return: batch_size x input_seq_len x d_model
        """

        for encoder_layer in self.encoder_layers:
            encoded_input_sequence = encoder_layer(
                encoded_input=encoded_input_sequence,
                attention_mask=input_attention_mask,
            )

        return encoded_input_sequence


class TransformerEncoderLayer(nn.Module):

    def __init__(
            self,
            d_model: int,
            d_ff: int,
            h: int,
            d_k: int,
            d_v: int,
            p_drop: float,
    ):
        super().__init__()

        self.d_model = d_model
        self.d_ff = d_ff
        self.h = h
        self.d_k = d_k
        self.d_v = d_v
        self.p_drop = p_drop

        self.multi_head_attention = MultiHeadAttention(d_model, h, d_k, d_v, mask_back_connections=False)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.layer_norm_1 = nn.LayerNorm(d_model)
        self.layer_norm_2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=p_drop)

    def forward(self, encoded_input: torch.FloatTensor, attention_mask: torch.BoolTensor) -> torch.FloatTensor:
        """
        :param encoded_input: batch_size x input_seq_len x d_model
        :param attention_mask: batch_size x input_seq_len
        :return: batch_size x input_seq_len x d_model
        """

        # Apply self-attention
        self_attention = self.multi_head_attention(
                encoded_input,
                encoded_input,
                encoded_input,
                attention_mask,
            )

        self_attention = self.dropout(self_attention)
        encoded_input = self.layer_norm_1(encoded_input + self_attention)

        # Apply feed-forward layers
        feed_forward = self.feed_forward(encoded_input)
        feed_forward = self.dropout(feed_forward)

        encoded_input = self.layer_norm_2(encoded_input + feed_forward)

        return encoded_input


class TransformerDecoder(nn.Module):
    def __init__(
            self,
            N: int,
            d_model: int,
            d_ff: int,
            h: int,
            d_k: int,
            d_v: int,
            p_drop: float
    ):
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
            for _ in range(N)
        ])

    def forward(
            self,
            input_sequence_encoding: torch.FloatTensor,
            input_attention_mask: torch.BoolTensor,
            output_sequence_encoding: torch.FloatTensor,
            output_attention_mask: torch.BoolTensor,
    ) -> torch.FloatTensor:
        for decoder_layer in self.decoder_layers:
            output_sequence_encoding = decoder_layer(
                input_sequence_encoding, input_attention_mask, output_sequence_encoding, output_attention_mask
            )
        return output_sequence_encoding


class TransformerDecoderLayer(nn.Module):
    def __init__(
            self,
            d_model: int,
            d_ff: int,
            h: int,
            d_k: int,
            d_v: int,
            p_drop: float
    ):
        super().__init__()

        self.d_model = d_model
        self.d_ff = d_ff
        self.h = h
        self.d_k = d_k
        self.d_v = d_v
        self.p_drop = p_drop

        self.multi_head_attention_1 = MultiHeadAttention(d_model, h, d_k, d_v, mask_back_connections=True)
        self.multi_head_attention_2 = MultiHeadAttention(d_model, h, d_k, d_v, mask_back_connections=False)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.layer_norm_1 = nn.LayerNorm(d_model)
        self.layer_norm_2 = nn.LayerNorm(d_model)
        self.layer_norm_3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=p_drop)

    def forward(
            self,
            input_sequence_encoding: torch.FloatTensor,
            input_attention_mask: torch.BoolTensor,
            output_sequence_encoding: torch.FloatTensor,
            output_attention_mask: torch.BoolTensor,
    ) -> torch.FloatTensor:

        # Apply self-attention
        self_attention = self.multi_head_attention_1(
            queries=output_sequence_encoding,
            keys=output_sequence_encoding,
            values=output_sequence_encoding,
            keys_attention_mask=output_attention_mask,
        )

        self_attention = self.dropout(self_attention)
        output_sequence_encoding = self.layer_norm_1(output_sequence_encoding + self_attention)

        # Apply cross-attention over encoder output
        cross_attention = self.multi_head_attention_2(
            queries=output_sequence_encoding,
            keys=input_sequence_encoding,
            values=input_sequence_encoding,
            keys_attention_mask=input_attention_mask,
        )
        cross_attention = self.dropout(cross_attention)
        output_sequence_encoding = self.layer_norm_2(output_sequence_encoding + cross_attention)

        # Apply feed-forward layers
        feed_forward = self.feed_forward(output_sequence_encoding)
        feed_forward = self.dropout(feed_forward)
        output_sequence_encoding = self.layer_norm_3(output_sequence_encoding + feed_forward)

        return output_sequence_encoding


class MultiHeadAttention(nn.Module):
    def __init__(
            self,
            d_model: int,
            h: int,
            d_k: int,
            d_v: int,
            mask_back_connections: bool
    ):
        super().__init__()
        self.h = h
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.mask_back_connections = mask_back_connections

        self.queries_transform = nn.Parameter(torch.empty(h, 1, d_model, d_k))
        self.keys_transform = nn.Parameter(torch.empty(h, 1, d_model, d_k))
        self.values_transform = nn.Parameter(torch.empty(h, 1, d_model, d_v))

        nn.init.xavier_uniform_(self.queries_transform)
        nn.init.xavier_uniform_(self.keys_transform)
        nn.init.xavier_uniform_(self.values_transform)

        self.output_transform = nn.Parameter(torch.empty(h * d_v, d_model))
        nn.init.xavier_uniform_(self.output_transform)

    def forward(
            self,
            queries: torch.FloatTensor,
            keys: torch.FloatTensor,
            values: torch.FloatTensor,
            keys_attention_mask: torch.BoolTensor,
    ) -> torch.FloatTensor:
        """
        :param queries: batch_size x queries_len x d_model
        :param keys: batch_size x keys_len x d_model
        :param values: batch_size x keys_len x d_model
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

        queries = torch.matmul(queries, self.queries_transform)
        keys = torch.matmul(keys, self.keys_transform)
        values = torch.matmul(values, self.values_transform)

        # Calculate attention weights

        attention_weights = torch.matmul(
            queries,
            torch.transpose(keys, -1, -2)
        ) / math.sqrt(self.d_k)

        # Mask attention
        self._mask_attention(attention_weights, keys_attention_mask)

        attention_weights = torch.softmax(attention_weights, -1)

        # Calculate attention outputs
        attention_head_outputs = torch.matmul(attention_weights, values)
        attention_head_outputs = attention_head_outputs.permute(1, 2, 0, 3)

        # Concatenate attention outputs and project back to d_model
        attention_head_outputs = attention_head_outputs.reshape(batch_size, queries_len, -1)
        result = torch.matmul(attention_head_outputs, self.output_transform)

        return result

    def _mask_attention(
            self,
            attention_weights: torch.FloatTensor,
            keys_attention_mask: torch.BoolTensor,
    ) -> None:
        """
        For all False values in keys_attention_mask set corresponding values in attention to -inf.
        If need to mask backwards connections set all attention values above diagonal to -inf.

        :param attention_weights: attention weights (before softmax) to update:
            h x batch_size x queries_len x keys_len
        :param keys_attention_mask: boolean tensor indicating non-PAD tokens in keys: batch_size x keys_len
        :return: None
        """

        h = attention_weights.shape[0]
        batch_size = attention_weights.shape[1]
        queries_len = attention_weights.shape[2]
        keys_len = attention_weights.shape[3]
        device = keys_attention_mask.device

        set_to_minus_inf = torch.full(
            (1, batch_size, queries_len, keys_len),
            False,
            dtype=torch.bool,
            device=device,
        )

        keys_attention_mask = ~(
            keys_attention_mask
            .view(batch_size, 1, keys_len)
            .expand(batch_size, queries_len, keys_len)
        )

        set_to_minus_inf = torch.logical_or(
            set_to_minus_inf,
            keys_attention_mask,
        )

        if self.mask_back_connections:
            indices = torch.triu_indices(queries_len, keys_len, 1)
            set_to_minus_inf[:, :, indices[0], indices[1]] = True

        attention_weights[
            set_to_minus_inf.expand(h, batch_size, queries_len, keys_len)
        ] = -math.inf


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        x = self.linear_1(x)
        x = torch.relu(x)
        x = self.linear_2(x)
        return x


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_len: int):
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

    def forward(self, embedded_sequence: torch.FloatTensor) -> torch.FloatTensor:
        batch_size = embedded_sequence.shape[0]
        sequence_length = embedded_sequence.shape[1]
        positional_encoding = self.positional_encodings[:sequence_length, :]
        positional_encoding = positional_encoding.unsqueeze(0).repeat(batch_size, 1, 1)
        return embedded_sequence + positional_encoding
