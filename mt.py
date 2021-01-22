import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import torchtext

import spacy
import numpy as np
import sh

import random
import math
import time

from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator
from model.transformer import Seq2Seq, Encoder, Decoder
from absl import app, flags, logging

sh.rm('-r', '-f', 'logs')
sh.mkdir('logs')

flags.DEFINE_integer('input_dim', 7854, '')
flags.DEFINE_integer('output_dim', 5972, '')
flags.DEFINE_integer('src_pad_idx', 1, '')
flags.DEFINE_integer('trg_pad_idx', 1, '')

flags.DEFINE_integer('hid_dim', 256, '')
flags.DEFINE_integer('enc_layers', 3, '')
flags.DEFINE_integer('dec_layers', 3, '')
flags.DEFINE_integer('enc_heads', 8, '')
flags.DEFINE_integer('dec_heads', 8, '')
flags.DEFINE_integer('enc_pf_dim', 512, '')
flags.DEFINE_integer('dec_pf_dim', 512, '')
flags.DEFINE_float('enc_dropout', .1, '')
flags.DEFINE_float('dec_dropout', .1, '')

flags.DEFINE_integer('batch_size', 128, '')
flags.DEFINE_boolean('debug', False, '')
flags.DEFINE_integer('epochs', 100, '')
flags.DEFINE_float('lr', 0.1, '')
flags.DEFINE_float('clip', .5, '')
FLAGS = flags.FLAGS


class MachineTranslator(pl.LightningModule):
    def __init__(self):
        super().__init__()
        transfomer_encoder = Encoder(
            input_dim=FLAGS.input_dim, 
            hid_dim=FLAGS.hid_dim, 
            n_layers=FLAGS.enc_layers, 
            n_heads=FLAGS.enc_heads, 
            pf_dim=FLAGS.enc_pf_dim, 
            dropout=FLAGS.enc_dropout,
            max_length=100)
        transfomer_decoder = Decoder(
            output_dim=FLAGS.output_dim, 
            hid_dim=FLAGS.hid_dim, 
            n_layers=FLAGS.dec_layers, 
            n_heads=FLAGS.dec_heads, 
            pf_dim=FLAGS.dec_pf_dim, 
            dropout=FLAGS.dec_dropout,
            max_length=100)

        self.model = Seq2Seq(
           encoder=transfomer_encoder, 
           decoder=transfomer_decoder,
           src_pad_idx=FLAGS.src_pad_idx, 
           trg_pad_idx=FLAGS.trg_pad_idx,
        )
        self.model.apply(self.initialize_weights)

        self.loss = torch.nn.CrossEntropyLoss(ignore_index=FLAGS.trg_pad_idx)

    def initialize_weights(self, m):
        if hasattr(m, 'weight') and m.weight.dim() > 1:
            nn.init.xavier_uniform_(m.weight.data)

    def prepare_data(self):
        spacy_de = spacy.load('de')
        spacy_en = spacy.load('en')

        def _tokenize_de(text):
            return [tok.text for tok in spacy_de.tokenizer(text)]

        def _tokenize_en(text):
            return [tok.text for tok in spacy_en.tokenizer(text)]

        SRC = Field(
            tokenize=_tokenize_de,
            init_token='<sos>',
            eos_token='<eos>',
            lower=True,
            batch_first=True
        )
        TRG = Field(
            tokenize=_tokenize_de,
            init_token='<sos>',
            eos_token='<eos>',
            lower=True,
            batch_first=True
        )
        
        self.train_data, self.valid_data, self.test_data = Multi30k.splits(
            exts=('.de','.en'),
            fields=(SRC, TRG)
        )
        
        SRC.build_vocab(self.train_data, min_freq=2)
        TRG.build_vocab(self.train_data, min_freq=2)
        
        self.input_dim = len(SRC.vocab)
        self.output_dim = len(TRG.vocab)

    def forward(self, src, trg):
        logits, _ = self.model(src, trg)
        return logits

    def training_step(self, batch, batch_idx):
        src = batch.src
        trg = batch.trg

        output = self.forward(src, trg[:, :-1])
        output_dim = output.shape[-1]

        output = output.contiguous().view(-1, output_dim) # (bsz x trg_seq_len-1) x output_dim
        trg = trg[:, 1:].contiguous().view(-1) # (bsz x (trg_seq_len-1))
        
        loss = self.loss(output, trg)
        ppl = math.exp(loss)

        return {
            'loss': loss,  
            'ppl': ppl,
            'log': {'train_loss': loss, 'train_ppl': ppl}
        }

    def validation_step(self, batch, batch_idx):
        src = batch.src
        trg = batch.trg

        output = self.forward(src, trg[:, :-1])
        output_dim = output.shape[-1]

        output = output.contiguous().view(-1, output_dim) # (bsz x trg_seq_len-1) x output_dim
        trg = trg[:, 1:].contiguous().view(-1) # (bsz x (trg_seq_len-1))
        
        loss = self.loss(output, trg)
        ppl = math.exp(loss)

        return {
            'loss': loss,  
            'ppl': ppl,
            'log': {'valid_loss': loss, 'valid_ppl': ppl}
        }

    def train_dataloader(self):
        return BucketIterator(
            dataset=self.train_data,
            batch_size=FLAGS.batch_size,
        )

    def val_dataloader(self):
        return BucketIterator(
            dataset=self.valid_data,
            batch_size=FLAGS.batch_size,
        )

    def test_dataloader(self):
        return BucketIterator(
            dataset=self.test_data,
            batch_size=FLAGS.batch_size,
        )

    def configure_optimizers(self):
        return torch.optim.SGD(
            self.parameters(),
            lr=FLAGS.lr,
        )


def main(_):
    model = MachineTranslator()
    trainer = pl.Trainer(
        default_root_dir='logs',
        gpus=(1 if torch.cuda.is_available() else 0),
        max_epochs=FLAGS.epochs,
        fast_dev_run=FLAGS.debug,
        gradient_clip_val=FLAGS.clip,
        logger=pl.loggers.TensorBoardLogger('logs/', name='mt', version=0),
    )
    trainer.fit(model)

if __name__ == '__main__':
    app.run(main)
