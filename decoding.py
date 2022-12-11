import torch
import torch.nn.functional as F
from sacremoses import MosesDetokenizer, MosesPunctNormalizer
from tokenizers import Tokenizer

from tqdm.auto import tqdm
from data import SpecialTokens
from model import TranslationModel

# it's a surprise tool that will help you later
detok = MosesDetokenizer(lang="en")
mpn = MosesPunctNormalizer()


def generate_square_subsequent_mask(sz, device):
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(src, tgt, device):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len, device)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=device).type(torch.bool)

    src_padding_mask = (src == 1).transpose(0, 1)
    tgt_padding_mask = (tgt == 1).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


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
    src = torch.nn.utils.rnn.pad_sequence(src, padding_value=1).to(device)
    tgt = torch.ones(1, src.size(1)).fill_(2).type(torch.long).to(device)
    time = torch.ones(src.size(1)).fill_(max_len).type(torch.long).to(device)

    src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt, device)
    memory = model.encode(src, src_mask, src_padding_mask)

    for i in range(max_len-1):
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt, device)

        out = model.decode(tgt, memory, tgt_mask, None, tgt_padding_mask, src_padding_mask)
        prob = model.generator(out[-1:, :])

        _, next_words = torch.max(prob, dim=-1)
        tgt = torch.cat([tgt, next_words], dim=0)

        mask = (next_words[0] == 3) & (time == max_len)
        time[mask] = i + 1
    return tgt, time


def _beam_search_decode(
    model: TranslationModel,
    src: torch.Tensor,
    max_len: int,
    tgt_tokenizer: Tokenizer,
    device: torch.device,
    beam_size: int,
    beam_spread: int,
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
    src = src.unsqueeze(1).to(device)
    tgt = torch.ones(1, src.size(1)).fill_(2).type(torch.long).to(device)

    src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt, device)
    memory = model.encode(src, src_mask, src_padding_mask)

    finished_beams = []
    beams = [(tgt, 0.0)]

    for i in range(max_len-1):
        new_beams = []

        for tgt, base_prob in beams:
            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt, device)
            out = model.decode(tgt, memory, tgt_mask, None, tgt_padding_mask, src_padding_mask)
            prob = model.generator(out[-1:, :])
            values, indices = torch.topk(F.log_softmax(prob[0, 0], dim=-1), beam_spread, dim=-1)
            for val, idx in zip(values, indices):
                new_beam = (torch.cat([tgt, idx.unsqueeze(0).unsqueeze(0)], dim=0), base_prob + val)
                if idx == 3:
                    finished_beams.append(new_beam)
                else:
                    new_beams.append(new_beam)

        beam_size -= len(finished_beams)
        if beam_size <= 0:
            break
        beams = sorted(new_beams, key=lambda x:x[1])[-beam_size:]

    if not finished_beams:
        return beams[-1][0]
    return sorted(finished_beams, key=lambda x:x[1])[-1][0]


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
            ''.join([SpecialTokens.BEGINNING.value, sentence.strip(), SpecialTokens.END.value])
            for sentence in src_sentences
        ])
    ]
    max_len = max([len(s) for s in srcs]) + 5
    # print(f'{max_len=}')

    sentences = []

    if translation_mode == 'greedy':
        # tgt_tokens, time = _greedy_decode(model, srcs, max_len, None, device=device)
        # for tokens, t in zip(tgt_tokens.T, time):
        #     toks = tokens[1:t].tolist()
        #     decoded = [tgt_tokenizer.id_to_token(id) for id in toks]
        #     sentences.append(detok.detokenize(decoded).replace(" '", "'"))
        batch_size = 64
        for batch_idx in tqdm(range((len(srcs) + batch_size - 1) // batch_size)):
            batch = srcs[batch_idx * batch_size:(1 + batch_idx) * batch_size]
            tgt_tokens, time = _greedy_decode(model, batch, max_len, None, device=device)
            for tokens, t in zip(tgt_tokens.T, time):
                toks = tokens[1:t].tolist()
                decoded = [tgt_tokenizer.id_to_token(id) for id in toks]
                sentences.append(detok.detokenize(decoded).replace(" '", "'"))
    elif translation_mode == 'beam':
        for src in tqdm(srcs):
            toks = _beam_search_decode(model, src, max_len, None, device, 5, 5).flatten().tolist()[1:]
            toks = toks[:toks.index(3)]
            decoded = [tgt_tokenizer.id_to_token(id) for id in toks]
            sentences.append(detok.detokenize(decoded).replace(" '", "'"))
    else:
        raise NotImplementedError()

    return sentences
