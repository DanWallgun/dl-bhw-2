import torch
from sacremoses import MosesDetokenizer, MosesPunctNormalizer
from tokenizers import Tokenizer

from data import SpecialTokens
from model import TranslationModel

# it's a surprise tool that will help you later
detok = MosesDetokenizer(lang="en")
mpn = MosesPunctNormalizer()

def generate_square_subsequent_mask(sz, device):
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def _greedy_decode(
    model: TranslationModel,
    src: torch.Tensor,
    max_len: int,
    tgt_tokenizer: Tokenizer,
    device: torch.device,
) -> torch.Tensor:
    """
    Given a batch of source sequences, predict its translations with greedy search.
    The decoding procedure terminates once either max_len steps have passed
    or the "end of sequence" token has been reached for all sentences in the batch.
    :param model: the model to use for translation
    :param src: a (batch, time) tensor of source sentence tokens
    :param max_len: the maximum length of predictions
    :param tgt_tokenizer: target language tokenizer
    :param device: device that the model runs on
    :return: a (batch, time) tensor with predictions
    """

    src_padded = torch.nn.utils.rnn.pad_sequence(src, padding_value=1)
    src_seq_len = src_padded.shape[0]
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=device).type(torch.bool)
    # src_padding_mask = (src_padded == 1).transpose(0, 1)

    src_padded = src_padded.to(device)
    src_mask = src_mask.to(device)

    memory = model.encode(src_padded, src_mask)
    ys = torch.ones(1, src_padded.size(1)).fill_(2).type(torch.long).to(device)
    tm = torch.ones(src_padded.size(1)).fill_(max_len).type(torch.long).to(device)
    for i in range(max_len-1):
        memory = memory.to(device)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0), device)
                    .type(torch.bool)).to(device)
        # print(ys.shape, memory.shape, tgt_mask.shape)
        out = model.decode(ys, memory, tgt_mask)
        # out = out.transpose(0, 1)
        # print(out.shape)
        prob = model.generator(out[-1:, :])
        # print(prob.shape)
        _, next_words = torch.max(prob, dim=-1)
        # print(next_words)
        ys = torch.cat([ys, next_words], dim=0)
        mask = (next_words[0] == 3) & (tm == max_len)
        # print(mask)
        tm[mask] = i
    # print(ys, tm)
    return ys, tm


def _beam_search_decode(
    model: TranslationModel,
    src: torch.Tensor,
    max_len: int,
    tgt_tokenizer: Tokenizer,
    device: torch.device,
    beam_size: int,
) -> torch.Tensor:
    """
    Given a batch of source sequences, predict its translations with beam search.
    The decoding procedure terminates once max_len steps have passed.
    :param model: the model to use for translation
    :param src: a (batch, time) tensor of source sentence tokens
    :param max_len: the maximum length of predictions
    :param tgt_tokenizer: target language tokenizer
    :param device: device that the model runs on
    :param beam_size: the number of hypotheses
    :return: a (batch, time) tensor with predictions
    """
    pass


@torch.inference_mode()
def translate(
    model: torch.nn.Module,
    src_sentences: list[str],
    src_tokenizer: Tokenizer,
    tgt_tokenizer: Tokenizer,
    translation_mode: str,
    device: torch.device,
) -> list[str]:
    """
    Given a list of sentences, generate their translations.
    :param model: the model to use for translation
    :param src_sentences: untokenized source sentences
    :param src_tokenizer: source language tokenizer
    :param tgt_tokenizer: target language tokenizer
    :param translation_mode: either "greedy", "beam" or anything more advanced
    :param device: device that the model runs on
    """

    model.eval()
    srcs = [
        torch.tensor(_.ids)
        for _ in src_tokenizer.encode_batch([
            ' '.join([SpecialTokens.BEGINNING.value, sentence.strip(), SpecialTokens.END.value])
            for sentence in src_sentences
        ])
    ]
    max_len = max([len(s) for s in srcs]) + 5
    if translation_mode == 'greedy':
        tgt_tokens, time = _greedy_decode(model, srcs, max_len, tgt_tokenizer, device=device)
    elif translation_mode == 'beam':
        tgt_tokens, time = _greedy_decode(model, srcs, max_len, tgt_tokenizer, device=device)
    else:
        raise NotImplementedError()
    # print(tgt_tokens.T, time)
    tgt_tokens = [
        tokens[:t].tolist()
        for tokens, t in zip(tgt_tokens.T, time)
    ]
    return tgt_tokenizer.decode_batch(tgt_tokens)
