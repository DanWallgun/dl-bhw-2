import torch
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

    # last_pos = torch.empty(src.shape[0]).fill_(max_len).type(torch.long).to(device)
    # tgt = torch.empty(1, src.shape[1]).fill_(2).type(torch.long).to(device)

    # src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt, device)
    # print(src.size(), tgt.size())
    # print(src_mask.size(), tgt_mask.size(), src_padding_mask.size(), tgt_padding_mask.size())
    # print(src_padding_mask)

    ys = torch.ones(1, src.size(1)).fill_(2).type(torch.long).to(device)
    tm = torch.ones(src.size(1)).fill_(max_len).type(torch.long).to(device)

    src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, ys, device)
    # print(src.size(), ys.size())
    # print(src_mask.size(), tgt_mask.size(), src_padding_mask.size(), tgt_padding_mask.size())

    memory = model.encode(src, src_mask, src_padding_mask)
    # print(memory.size())

    for i in range(max_len-1):
        memory = memory.to(device)
        # tgt_mask = (generate_square_subsequent_mask(ys.size(0), device)
        #             .type(torch.bool)).to(device)
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, ys, device)

        # print('mem ys', memory.size(), ys.size())
        # print('masks', src_mask.size(), tgt_mask.size(), src_padding_mask.size(), tgt_padding_mask.size())

        out = model.decode(ys, memory, tgt_mask, None, tgt_padding_mask, src_padding_mask)
        # out = out.transpose(0, 1)
        # print(out.shape)
        prob = model.generator(out[-1:, :])
        # print(prob.shape)
        _, next_words = torch.max(prob, dim=-1)
        # print(next_words)
        ys = torch.cat([ys, next_words], dim=0)
        mask = (next_words[0] == 3) & (tm == max_len)
        # print(mask)
        tm[mask] = i + 1
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
            ''.join([SpecialTokens.BEGINNING.value, sentence.strip(), SpecialTokens.END.value])
            for sentence in src_sentences
        ])
    ]
    max_len = max([len(s) for s in srcs]) + 5
    if translation_mode == 'greedy':
        tgt_tokens, time = _greedy_decode(model, srcs, max_len, tgt_tokenizer, device=device)
        # tgt_tokens__time = []
        # for src in tqdm(srcs):
        #     tgt_tokens__time.append(_greedy_decode(model, [src], max_len, tgt_tokenizer, device))
    elif translation_mode == 'beam':
        raise NotImplementedError()
    else:
        raise NotImplementedError()
    # print(tgt_tokens.T, time)
    # out_tokens = [
    #     tokens[:t+1].flatten().tolist()
    #     # for tokens, t in zip(tgt_tokens.T, time)
    #     for tokens, t in tgt_tokens__time
    # ]
    out_tokens = []
    sentences = []
    for tokens, t in zip(tgt_tokens.T, time):
        toks = tokens[1:t].tolist()
        # assert len(toks) == max_len
        # for idx in range(max_len):
        #     if toks[idx] == 3:
        #         toks = toks[:idx+1]
        #         break
        # print(tokens, idx, t)
        # assert idx == t
        # out_tokens.append(toks)
        sentences.append(detok.detokenize([tgt_tokenizer.id_to_token(id) for id in toks]).replace(" '", "'"))

    # sentences = tgt_tokenizer.decode_batch(out_tokens)
    # return fix_sentences(sentences)
    return sentences


def dumb_greedy_decode(model, src, max_len, device):
    src = src.view(-1, 1).to(device)
    num_tokens = src.shape[0]
    max_len = max_len or (num_tokens + 5)
    src_mask = (torch.zeros(num_tokens, num_tokens, device=device)).type(torch.bool)

    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(2).type(torch.long).to(device)
    for i in range(max_len-1):
        memory = memory.to(device)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0), device)
                    .type(torch.bool)).to(device)
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == 3:
            break
    return ys


@torch.inference_mode()
def dumb_translate(
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
    if translation_mode == 'greedy':
        tgt_tokens = []
        for src in tqdm(srcs):
            tgt_tokens.append(dumb_greedy_decode(model, src, max_len, device).flatten().tolist())
    elif translation_mode == 'beam':
        raise NotImplementedError()
    else:
        raise NotImplementedError()
    # print(tgt_tokens.T, time)
    sentences = tgt_tokenizer.decode_batch(tgt_tokens)
    return fix_sentences(sentences)


def fix_sentences(sentences):
    fixed_sentences = []
    for s in sentences:
        fs = s.replace(' ,', ',').replace(" ' ", "'")
        if fs[-2:] == ' .':
            fs = fs[:-2] + '.'
        fixed_sentences.append(fs)
    return fixed_sentences