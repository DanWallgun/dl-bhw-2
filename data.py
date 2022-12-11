import os
from enum import Enum
from pathlib import Path

import torch
import xml.etree.ElementTree as ET
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from torch.utils.data import Dataset


def process_training_file(input_path: Path, output_path: Path):
    """
    Processes raw training files ("train.tags.SRC-TGT.*"), saving the output as a sequence of unformatted examples
    (.txt file, one example per line).
    :param input_path: Path to the file with the input data (formatted examples)
    :param output_path: Path to the file with the output data (one example per line)
    """
    tree = ET.fromstring('<root>' + open(input_path).read() + '</root>')
    with open(output_path, 'w') as f:
        for doc in tree:
            f.write(doc.find('description').tail.strip() + '\n')



def process_evaluation_file(input_path: Path, output_path: Path):
    """
    Processes raw validation and testing files ("IWSLT17.TED.{dev,test}2010.SRC-TGT.*.xml"),
    saving the output as a sequence of unformatted examples (.txt file, one example per line).
    :param input_path: Path to the file with the input data (formatted examples)
    :param output_path: Path to the file with the output data (one example per line)
    """
    tree = ET.parse(input_path)
    with open(output_path, 'w') as f:
        for seg in tree.findall('.//seg'):
            f.write(seg.text.strip() + '\n')


def convert_files(base_path: Path, output_path: Path):
    """
    Given a directory containing all the dataset files, convert each one into the "one example per line" format.
    :param base_path: Path containing files with original data
    :param output_path: Path containing files with processed data
    """

    for language in "de", "en":
        process_training_file(
            base_path / f"train.tags.de-en.{language}",
            output_path / f"train.{language}.txt",
        )
        process_evaluation_file(
            base_path / f"IWSLT17.TED.dev2010.de-en.{language}.xml",
            output_path / f"val.{language}.txt",
        )
        process_evaluation_file(
            base_path / f"IWSLT17.TED.tst2010.de-en.{language}.xml",
            output_path / f"test.{language}.txt",
        )


class TranslationDataset(Dataset):
    def __init__(
        self,
        src_file_path,
        tgt_file_path,
        src_tokenizer: Tokenizer,
        tgt_tokenizer: Tokenizer,
        max_len=32,
    ):
        """
        Loads the training dataset and parses it into separate tokenized training examples.
        No padding should be applied at this stage
        :param src_file_path: Path to the source language training data
        :param tgt_file_path: Path to the target language training data
        :param src_tokenizer: Trained tokenizer for the source language
        :param tgt_tokenizer: Trained tokenizer for the target language
        :param max_len: Maximum length of source and target sentences for each example:
        if either of the parts contains more tokens, it needs to be filtered.
        """
        # your code here
        # self.src_tokenizer = Tokenizer.from_file(src_tokenizer)
        # self.tgt_tokenizer = Tokenizer.from_file(tgt_tokenizer)
        # self.src_tokenizer = src_tokenizer
        # self.tgt_tokenizer = tgt_tokenizer

        raw_src = [
            ''.join([SpecialTokens.BEGINNING.value, line.strip(), SpecialTokens.END.value])
            for line in open(src_file_path, 'r')
        ]
        raw_tgt = [
            ''.join([SpecialTokens.BEGINNING.value, line.strip(), SpecialTokens.END.value])
            for line in open(tgt_file_path, 'r')
        ]

        src = [torch.tensor(_.ids) for _ in src_tokenizer.encode_batch(raw_src)]
        tgt = [torch.tensor(_.ids) for _ in tgt_tokenizer.encode_batch(raw_tgt)]

        self.src, self.tgt = [], []

        for s, t in zip(src, tgt):
            if len(s) > max_len or len(t) > max_len:
                continue
            self.src.append(s)
            self.tgt.append(t)

    def __len__(self):
        return len(self.src)

    def __getitem__(self, i):
        return (self.src[i], self.tgt[i])

    def collate_translation_data(self, batch):
        """
        Given a batch of examples with varying length, collate it into `source` and `target` tensors for the model.
        This method is meant to be used when instantiating the DataLoader class for training and validation datasets in your pipeline.
        """
        srcs, tgts = [], []
        for src, tgt in batch:
            srcs.append(src)
            tgts.append(tgt)
        return (
            torch.nn.utils.rnn.pad_sequence(srcs, padding_value=1),
            torch.nn.utils.rnn.pad_sequence(tgts, padding_value=1)
        )



class SpecialTokens(Enum):
    UNKNOWN = "[UNK]"
    PADDING = "[PAD]"
    BEGINNING = "[BOS]"
    END = "[EOS]"


def train_tokenizers(base_dir: Path, save_dir: Path):
    """
    Trains tokenizers for source and target languages and saves them to `save_dir`.
    :param base_dir: Directory containing processed training and validation data (.txt files from `convert_files`)
    :param save_dir: Directory for storing trained tokenizer data (two files: `tokenizer_de.json` and `tokenizer_en.json`)
    """
    trainer = BpeTrainer(vocab_size=30000, special_tokens=[t.value for t in SpecialTokens])

    en_tokenizer = Tokenizer(BPE(unk_token=SpecialTokens.UNKNOWN.value))
    en_tokenizer.pre_tokenizer = Whitespace()
    en_tokenizer.train(files=[os.path.join(base_dir, name) for name in os.listdir(base_dir) if name.endswith('en.txt')], trainer=trainer)
    en_tokenizer.save(str(save_dir / "tokenizer_en.json"))

    de_tokenizer = Tokenizer(BPE(unk_token=SpecialTokens.UNKNOWN.value))
    de_tokenizer.pre_tokenizer = Whitespace()
    de_tokenizer.train(files=[os.path.join(base_dir, name) for name in os.listdir(base_dir) if name.endswith('de.txt')], trainer=trainer)
    de_tokenizer.save(str(save_dir / "tokenizer_de.json"))
