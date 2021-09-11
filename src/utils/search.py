'''Beam-search and greedy search sequence decoding.
Majorly inspired by https://github.com/joeynmt/joeynmt/blob/master/joeynmt/search.py
'''

import torch as t


def beam_search_decode(model, encoded_source, source_attention_masks, beam_size=5, max_len=120, alpha=-1):

    device = encoded_source.device

    start_batch_size = source_attention_masks.size(0)

    # encoded_source: start_batch_size * beam_size x h_model
    encoded_source = tile(encoded_source, beam_size, dim=0)
    # source_attention_masks: start_batch_size * beam_size x 1
    source_attention_masks = tile(source_attention_masks, beam_size, dim=0)

    hypotheses_attention_masks = t.full(
        (start_batch_size * beam_size, max_len),
        True,
        dtype=t.bool,
        device=device
    )

    hypotheses_input_ids = t.full(
        (start_batch_size * beam_size, 1),
        1,  # START TOKEN ID
        dtype=t.long,
        device=device
    )

    hypotheses_log_probas = t.zeros((start_batch_size, beam_size), dtype=t.float, device=device)
    hypotheses_log_probas[:, 1:] = float('-inf')  # Only first hypotheses in each batch in the beginning

    batch_enumeration = t.arange(start_batch_size, dtype=t.long, device=device)

    beam_offset = t.arange(
        0,
        start_batch_size * beam_size,
        step=beam_size,
        dtype=t.long,
        device=device
    )

    result = [None for _ in range(start_batch_size)]

    for step in range(max_len):

        # current_batch_size is equal to start_batch_size until we terminate one of the sequences
        current_batch_size = hypotheses_log_probas.size(0)
        with t.no_grad():
            # current_batch_size * beam_size x vocab_size
            logits = model.decoder_function(
                encoded_source,
                source_attention_masks,
                hypotheses_input_ids,
                hypotheses_attention_masks[:current_batch_size * beam_size, :step + 1]
            )[:, -1]

        vocab_size = logits.size(-1)

        # last_token_log_probas: current_batch_size * beam_size x 1 x vocab_size
        # last_token_log_probas.squeeze(1): current_batch_size * beam_size x vocab_size
        # hypotheses_log_probas: current_batch_size x beam_size
        # hypotheses_log_probas.view(-1).unsqueeze(-1): current_batch_size * beam_size x 1
        new_hypotheses_scores = (
                t.log_softmax(logits, dim=-1).squeeze(1) +
                hypotheses_log_probas.view(-1).unsqueeze(1)
        )

        new_hypotheses_scores = new_hypotheses_scores.reshape(current_batch_size, beam_size * vocab_size)

        if alpha > -1:
            length_penalty = ((5.0 + (step + 1)) / 6.0) ** alpha
            new_hypotheses_scores /= length_penalty

        # topk_scores: current_batch_size x beam_size
        # topk_ids: current_batch_size x beam_size
        top_hypotheses_scores_per_beam, top_hypotheses_indices_per_beam = new_hypotheses_scores.topk(
            beam_size, dim=-1
        )

        if alpha > -1:
            hypotheses_log_probas = top_hypotheses_scores_per_beam * length_penalty
        else:
            hypotheses_log_probas = top_hypotheses_scores_per_beam

        # top_hypotheses_to_continue: current_batch_size x beam_size
        # top_hypotheses_continuations: current_batch_size x beam_size
        top_hypotheses_to_continue = top_hypotheses_indices_per_beam.floor_divide(vocab_size)
        top_hypotheses_continuations = top_hypotheses_indices_per_beam.fmod(vocab_size)

        # beam_offset: current_batch_size
        # batch_index: current_batch_size x beam_size absolute indices of top hypotheses continuations
        batch_index = (
                top_hypotheses_to_continue
                + beam_offset[:top_hypotheses_to_continue.size(0)].unsqueeze(1)
        )

        # current_batch_size * beam_size absolute indices of hypotheses to continue
        # one hypotheses may be continued several times if those are more probable
        select_indices = batch_index.view(-1)

        # hypotheses_input_ids: current_batch_size * beam_size x step + 1
        hypotheses_input_ids = t.cat(
            [hypotheses_input_ids.index_select(0, select_indices), top_hypotheses_continuations.view(-1, 1)],
            -1
        )

        # is_finished: current_batch_size x beam_size
        is_finished = top_hypotheses_continuations.eq(2)  # END TOKEN ID

        if step + 1 == max_len:
            is_finished.fill_(True)

        # end_condition: current_batch_size
        end_condition = is_finished[:, 0].eq(True)

        if end_condition.any():
            predictions = hypotheses_input_ids.view(current_batch_size, beam_size, step + 2)
            finished_indices = end_condition.eq(True).nonzero(as_tuple=False).view(-1)
            unfinished_indices = end_condition.eq(False).nonzero(as_tuple=False).view(-1)

            for i in finished_indices:
                result[batch_enumeration[i]] = predictions[i][0].tolist()

            hypotheses_log_probas = hypotheses_log_probas.index_select(0, unfinished_indices)
            batch_index = batch_index.index_select(0, unfinished_indices)
            batch_enumeration = batch_enumeration.index_select(0, unfinished_indices)
            hypotheses_input_ids = predictions.index_select(0, unfinished_indices).view(-1, step + 2)
            if len(unfinished_indices) == 0:
                break

        select_indices = batch_index.view(-1)
        encoded_source = encoded_source.index_select(0, select_indices)
        source_attention_masks = source_attention_masks.index_select(0, select_indices)

    return result


def greedy_decode(self, model, source_encoding, source_attention_mask, max_len=128):
    pass


def tile(x, count, dim=0):
    """
    Tiles x on dimension dim count times. From OpenNMT. Used for beam search.
    :param x: tensor to tile
    :param count: number of tiles
    :param dim: dimension along which the tensor is tiled
    :return: tiled tensor
    """
    if isinstance(x, tuple):
        h, c = x
        return tile(h, count, dim=dim), tile(c, count, dim=dim)

    perm = list(range(len(x.size())))
    if dim != 0:
        perm[0], perm[dim] = perm[dim], perm[0]
        x = x.permute(perm).contiguous()
    out_size = list(x.size())
    out_size[0] *= count
    batch = x.size(0)
    x = x.view(batch, -1) \
        .transpose(0, 1) \
        .repeat(count, 1) \
        .transpose(0, 1) \
        .contiguous() \
        .view(*out_size)
    if dim != 0:
        x = x.permute(perm).contiguous()
    return x
