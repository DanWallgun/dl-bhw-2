import math

import torch
import torch.nn as nn
from torch import Tensor

'''
using https://pytorch.org/tutorials/beginner/translation_transformer.html#seq2seq-network-using-transformer
'''

class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


class TranslationModel(nn.Module):
    def __init__(
        self,
        num_encoder_layers: int,
        num_decoder_layers: int,
        emb_size: int,
        dim_feedforward: int,
        n_head: int,
        src_vocab_size: int,
        tgt_vocab_size: int,
        dropout_prob: float,
    ):
        """
        Creates a standard Transformer encoder-decoder model.
        :param num_encoder_layers: Number of encoder layers
        :param num_decoder_layers: Number of decoder layers
        :param emb_size: Size of intermediate vector representations for each token
        :param dim_feedforward: Size of intermediate representations in FFN layers
        :param n_head: Number of attention heads for each layer
        :param src_vocab_size: Number of tokens in the source language vocabulary
        :param tgt_vocab_size: Number of tokens in the target language vocabulary
        :param dropout_prob: Dropout probability throughout the model
        """
        super().__init__()
        # your code here
        self.transformer = nn.Transformer(d_model=emb_size,
                                       nhead=n_head,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout_prob)
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout_prob)

    def forward(
        self,
        src_tokens: Tensor, tgt_tokens: Tensor,
        src_mask: Tensor, tgt_mask: Tensor,
        src_padding_mask: Tensor, tgt_padding_mask: Tensor,
        memory_key_padding_mask: Tensor,
    ):
        """
        Given tokens from a batch of source and target sentences, predict logits for next tokens in target sentences.
        """
        src_emb = self.positional_encoding(self.src_tok_emb(src_tokens))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt_tokens))
        outs = self.transformer(src_emb, tgt_emb,
                                src_mask, tgt_mask, None,
                                src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor, src_key_padding_mask: Tensor = None):
        return self.transformer.encoder(
            self.positional_encoding(self.src_tok_emb(src)),
            src_mask,
            src_key_padding_mask=src_key_padding_mask
        )

    def decode(
        self, tgt: Tensor, memory: Tensor,
        tgt_mask: Tensor, memory_mask: Tensor = None,
        tgt_key_padding_mask: Tensor = None, memory_key_padding_mask: Tensor = None):
        return self.transformer.decoder(
            self.positional_encoding(self.tgt_tok_emb(tgt)),
            memory,
            tgt_mask,
            memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )