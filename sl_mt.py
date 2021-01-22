import math
import os

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import sh
import spacy
import torch
import torch.nn as nn
import torch.optim as optim
import torchtext
from absl import app, flags, logging
from torchtext.data import (BucketIterator, Dataset, Example, Field,
                            TabularDataset)

# from torchtext.data.metrics import bleu_score
from metric import bleu as bleu_score
from model.transformer import Decoder, Encoder, Seq2Seq

sh.rm('-r', '-f', 'logs')
sh.mkdir('logs')

flags.DEFINE_integer('input_dim', 1814, '')
flags.DEFINE_integer('output_dim', 734, '')
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

flags.DEFINE_integer('batch_size', 64, '')
flags.DEFINE_boolean('debug', False, '')
flags.DEFINE_integer('epochs', 300, '')
flags.DEFINE_float('lr', 0.01, '')
flags.DEFINE_float('clip', .5, '')
flags.DEFINE_float('momentum', .9, '')

flags.DEFINE_string('data_path', './annotation', '')
FLAGS = flags.FLAGS


class SignTransformer(pl.LightningModule):
    def __init__(self, data_dir, lr):
        super().__init__()
        
        self.data_dir = data_dir
        self.lr = lr

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

    def setup(self, stage=None):
        def _tokenize(_string):
            return _string.split(' ')

        self.TXT = Field(
            tokenize=_tokenize,
            init_token='<sos>',
            eos_token='<eos>',
            batch_first=True
        )
        self.GLS = Field(
            tokenize=_tokenize,
            init_token='<sos>',
            eos_token='<eos>',
            batch_first=True
        )  
        
        def _prepare_ds(mode):
            data_path = os.path.join(self.data_dir, f'PHOENIX-2014-T.{mode}.corpus.csv')
            ds = SignTranslationData(data_path, (self.TXT, self.GLS))
            return ds
        
        self.train_data, self.val_data, self.test_data = map(_prepare_ds, ('train', 'dev', 'test'))
        
        # build vocab
        self.TXT.build_vocab(self.train_data, min_freq=2)
        self.GLS.build_vocab(self.train_data, min_freq=2)

    def train_dataloader(self):
        return BucketIterator(
            dataset=self.train_data,
            batch_size=FLAGS.batch_size,
        )

    def val_dataloader(self):
        return BucketIterator(
            dataset=self.val_data,
            batch_size=FLAGS.batch_size,
        )

    def test_dataloader(self):
        return BucketIterator(
            dataset=self.test_data,
            batch_size=FLAGS.batch_size,
        )

    def initialize_weights(self, m):
        if hasattr(m, 'weight') and m.weight.dim() > 1:
            nn.init.xavier_uniform_(m.weight.data)

    def forward(self, txt, gls):
        output, _ = self.model(txt, gls)
        return output

    def training_step(self, batch, batch_idx):
        txt = batch.txt
        gls = batch.gls
        
        output = self.forward(txt, gls[:, :-1])
        output_dim = output.shape[-1]

        pred = output.contiguous().view(-1, output_dim)
        trg_gls = gls[:, 1:].contiguous().view(-1)

        loss = self.loss(pred, trg_gls)
        ppl = math.exp(loss)
        
        # get_blue
        _gls_hyp = self.get_gls_hyp(output)
        _gls_ref = self.get_gls_ref(gls)
        bleu = bleu_score(_gls_hyp, _gls_ref)
        
        pbar = {'ppl': ppl, 'bleu': bleu['bleu4']}
        
        return {
            'loss': loss,
            'progress_bar': pbar,
            'log': {
                'train_loss': loss, 
                'train_ppl': ppl,
            }
        }

    def validation_step(self, batch, batch_idx):
        txt = batch.txt
        gls = batch.gls
        
        output = self.forward(txt, gls[:, :-1])
        output_dim = output.shape[-1]

        pred = output.contiguous().view(-1, output_dim)
        trg_gls = gls[:, 1:].contiguous().view(-1)

        loss = self.loss(pred, trg_gls)
        ppl = math.exp(loss)
        
        # get_blue
        _gls_hyp = self.get_gls_hyp(output)
        _gls_ref = self.get_gls_ref(gls)
        bleu = bleu_score(_gls_hyp, _gls_ref)
        
        pbar = {'ppl': ppl, 'bleu': bleu['bleu4']}
        
        return {
            'loss': loss,
            'progress_bar': pbar,
            'log': {
                'valid_loss': loss, 
                'valid_ppl': ppl,
                'bleu1': bleu['bleu1'],
                'bleu2': bleu['bleu2'],
                'bleu3': bleu['bleu3'],
                'bleu4': bleu['bleu4']
            }
        }

    def test_step(self, batch, batch_idx):
        import IPython; IPython.embed(); exit(1)

    def get_gls_hyp(self, logits):
        logits = logits.argmax(2)
        bsz, seq_len = logits.shape
        pred_gls = []
        for i in range(bsz):
            gls_hyp = []
            for j in range(seq_len):
                if logits[i][j] == self.GLS.vocab['<eos>']:
                    break
                gls_hyp.append(self.GLS.vocab.itos[logits[i][j]])
            pred_gls.append(gls_hyp)
        return [' '.join(token) for token in pred_gls]

    def get_gls_ref(self, gls):
        gls_ref_list = []
        for gls_indicies in gls:
            gls_ref = []
            for _gls_token in gls_indicies[1:]:
                if _gls_token == self.GLS.vocab.stoi['<eos>'] or _gls_token == self.GLS.vocab.stoi['<pad>']:
                    break
                gls_ref.append(self.GLS.vocab.itos[_gls_token])
            gls_ref_list.append(gls_ref)
        return [' '.join(token) for token in gls_ref_list]

    def configure_optimizers(self):
        return torch.optim.SGD(
            self.parameters(),
            lr=self.lr,
            # momentum=FLAGS.momentum,
        )


class SignTranslationData(Dataset):
    def __init__(self, data_path: str, fields: tuple):
        df = pd.read_csv(data_path, usecols=['translation', 'orth'], delimiter='|')
        
        txt = df['translation'].tolist()
        gls = df['orth'].tolist()

        fields = [
            ('txt', fields[0]),
            ('gls', fields[1]),
        ]

        examples = self.get_examples(txt, gls, fields)
        super().__init__(examples, fields)

    def get_examples(self, txt, gls, fields):
        examples = []
        for _txt, _gls in zip(txt, gls):
            examples.append(
                Example.fromlist(
                    [_txt, _gls],
                    fields,
                )
            )

        return examples

        
def main(_):
    # ds = SignTranslationDataset(FLAGS.data_path)
    model = SignTransformer(FLAGS.data_path, FLAGS.lr)
    
    trainer = pl.Trainer(
        default_root_dir='logs',
        gpus=(1 if torch.cuda.is_available() else 0),
        max_epochs=FLAGS.epochs,
        fast_dev_run=FLAGS.debug,
        # gradient_clip_val=FLAGS.clip,
        auto_lr_find=True,
    )
    trainer.fit(model)
    # trainer.test()


if __name__ == "__main__":
    app.run(main)
