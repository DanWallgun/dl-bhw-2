import os
from argparse import ArgumentParser
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sacrebleu.metrics import BLEU
from tokenizers import Tokenizer
from tqdm import trange
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter

from data import TranslationDataset
from decoding import translate
from model import TranslationModel

os.environ["TOKENIZERS_PARALLELISM"] = "false"


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


def train_epoch(
    model: TranslationModel,
    train_dataloader,
    optimizer,
    scheduler,
    device,
    tbwriter=None,
    step=None,
):
    # train the model for one epoch
    # you can obviously add new arguments or change the API if it does not suit you
    model.train()
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=1)
    acc_loss = 0
    num_samples = 0
    pb = tqdm(train_dataloader)
    for idx, (src, tgt) in enumerate(pb):

        src = src.to(device)
        tgt = tgt.to(device)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, device)

        logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)

        optimizer.zero_grad()

        tgt_output = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_output.reshape(-1))
        loss.backward()

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        acc_loss += loss.item() * src.shape[1]
        num_samples += src.shape[1]
        pb.set_description(f'TrainLoss = {acc_loss/num_samples:.3f}')

        if tbwriter is not None:
            tbwriter.add_scalar('AccLoss/train', acc_loss/num_samples, step + idx)
            tbwriter.add_scalar('LearningRate', [pg['lr'] for pg in optimizer.param_groups][0], step + idx)

    return acc_loss / num_samples


@torch.inference_mode()
def evaluate(model: TranslationModel, val_dataloader, device):
    # compute the loss over the entire validation subset
    model.eval()
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=1)
    losses = 0
    for src, tgt in tqdm(val_dataloader):
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, device)

        logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

        tgt_output = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_output.reshape(-1))
        losses += loss.item()

    return losses / len(val_dataloader)


def train_model(data_dir, tokenizer_path, num_epochs):
    src_tokenizer = Tokenizer.from_file(str(tokenizer_path / "tokenizer_de.json"))
    tgt_tokenizer = Tokenizer.from_file(str(tokenizer_path / "tokenizer_en.json"))

    train_dataset = TranslationDataset(
        data_dir / "train.de.txt",
        data_dir / "train.en.txt",
        src_tokenizer,
        tgt_tokenizer,
        max_len=128,  # might be enough at first
    )
    val_dataset = TranslationDataset(
        data_dir / "val.de.txt",
        data_dir / "val.en.txt",
        src_tokenizer,
        tgt_tokenizer,
        max_len=128,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    model = TranslationModel(
        # your code here
        num_encoder_layers=3,
        num_decoder_layers=3,
        emb_size=512,
        dim_feedforward=512,
        n_head=8,
        src_vocab_size=src_tokenizer.get_vocab_size(),
        tgt_vocab_size=tgt_tokenizer.get_vocab_size(),
        dropout_prob=0.2,
    )
    model.to(device)
    model.load_state_dict(torch.load("checkpoint_last.pth"))

    print(f'#params = {sum([p.numel() for p in model.parameters()])}')

    # create loss, optimizer, scheduler objects, dataloaders etc.
    # don't forget about collate_fn
    # if you intend to use AMP, you might need something else
    batch_size = 128
    num_workers = 12
    train_dataloader = DataLoader(train_dataset, collate_fn=train_dataset.collate_translation_data, batch_size=batch_size, num_workers=num_workers)
    val_dataloader = DataLoader(val_dataset, collate_fn=val_dataset.collate_translation_data, batch_size=batch_size, num_workers=num_workers)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=1e-3,
        epochs=num_epochs, steps_per_epoch=len(train_dataloader),
        pct_start=0.05
    ) if num_epochs > 0 else None

    min_val_loss = float("inf")

    tbwriter = SummaryWriter()

    for epoch in trange(1, num_epochs + 1):
        step = (epoch - 1) * len(train_dataloader)

        train_loss = train_epoch(
            model, train_dataloader,
            optimizer, scheduler,
            device,
            tbwriter, step
        )
        val_loss = evaluate(model, val_dataloader, device)

        print(train_loss, val_loss)
        tbwriter.add_scalar('Loss/val', val_loss, step)

        # might be useful to translate some sentences from validation to check your decoding implementation

        # also, save the best checkpoint somewhere around here
        if val_loss < min_val_loss:
            print("New best loss! Saving checkpoint")
            torch.save(model.state_dict(), "checkpoint_best.pth")
            min_val_loss = val_loss

        # and the last one in case you need to recover
        # by the way, is this sufficient?
        torch.save(model.state_dict(), "checkpoint_last.pth")

    # load the best checkpoint
    model.load_state_dict(torch.load("checkpoint_best.pth"))
    return model


def translate_test_set(model: TranslationModel, data_dir, tokenizer_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    model.to(device)
    model.eval()
    src_tokenizer = Tokenizer.from_file(str(tokenizer_path / "tokenizer_de.json"))
    tgt_tokenizer = Tokenizer.from_file(str(tokenizer_path / "tokenizer_en.json"))

    greedy_translations = []
    with open(data_dir / "test.de.txt") as input_file, open(
        "answers_greedy.txt", "w+"
    ) as output_file:
        # translate with greedy search
        greedy_translations = translate(
            model,
            input_file.readlines(),
            src_tokenizer,
            tgt_tokenizer,
            'greedy',
            device
        )
        output_file.write('\n'.join(greedy_translations) + '\n')

    beam_translations = []
    with open(data_dir / "test.de.txt") as input_file, open(
        "answers_beam.txt", "w+"
    ) as output_file:
        # translate with beam search
        pass

    with open(data_dir / "test.en.txt") as input_file:
        references = [line.strip() for line in input_file]

    bleu = BLEU()
    bleu_greedy = bleu.corpus_score(greedy_translations, [references]).score

    # we're recreating the object, as it might cache some stats
    # bleu = BLEU()
    # bleu_beam = bleu.corpus_score(beam_translations, [references]).score
    bleu_beam = 0.0

    print(f"BLEU with greedy search: {bleu_greedy}, with beam search: {bleu_beam}")
    # maybe log to wandb/comet/neptune as well


if __name__ == "__main__":
    parser = ArgumentParser()
    data_group = parser.add_argument_group("Data paths")
    data_group.add_argument(
        "--data-dir", type=Path, help="Path to the directory containing processed data"
    )
    data_group.add_argument(
        "--tokenizer-path", type=Path, help="Path to the trained tokenizer files"
    )

    # argument groups are useful for separating semantically different parameters
    hparams_group = parser.add_argument_group("Training hyperparameters")
    hparams_group.add_argument(
        "--num-epochs", type=int, default=50, help="Number of training epochs"
    )

    args = parser.parse_args()

    model = train_model(args.data_dir, args.tokenizer_path, args.num_epochs)
    translate_test_set(model, args.data_dir, args.tokenizer_path)
