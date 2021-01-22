import math
import os
import random

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import sh
import spacy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchtext
from absl import app, flags, logging
from torchtext.data import (BucketIterator, Dataset, Example, Field,
                            TabularDataset)

from metric import bleu as bleu_score
from model.transformer import Decoder, Encoder, Seq2Seq
from onmt.decoders.transformer import TransformerDecoder
from onmt.encoders.transformer import TransformerEncoder
from onmt.models.model import NMTModel
from onmt.modules import Embeddings
from onmt.modules.util_class import Cast

flags.DEFINE_integer('input_dim', 1236, '')
flags.DEFINE_integer('output_dim', 2892, '')
flags.DEFINE_integer('src_pad_idx', 1, '')
flags.DEFINE_integer('trg_pad_idx', 1, '')

# model related
flags.DEFINE_string('model_name', 'tutorial', '')
flags.DEFINE_integer('num_layers', 2, '')
flags.DEFINE_integer('d_model', 512, '')
flags.DEFINE_integer('d_ff', 2048, '')
flags.DEFINE_integer('n_heads', 8, '')
flags.DEFINE_float('dropout', .1, '')

flags.DEFINE_integer('batch_size', 256, '')
flags.DEFINE_boolean('debug', False, '')
flags.DEFINE_integer('epochs', 100, '')
flags.DEFINE_float('lr', 0.01, '')
flags.DEFINE_float('clip', .5, '')
flags.DEFINE_float('momentum', .9, '')
flags.DEFINE_float('weight_decay', 1e-4, '')

flags.DEFINE_string('data_path', './data', '')
flags.DEFINE_string('save_log', 'gls2txt', '')
FLAGS = flags.FLAGS


def get_onmt_transformer():
    encoder = TransformerEncoder(
        num_layers=FLAGS.num_layers,
        d_model=FLAGS.d_model,
        heads=FLAGS.n_heads,
        d_ff=FLAGS.d_ff,
        dropout=FLAGS.dropout,
        embeddings=Embeddings(
            word_vec_size=FLAGS.d_model,
            word_vocab_size=FLAGS.input_dim,
            word_padding_idx=FLAGS.src_pad_idx,
            position_encoding=True,
            dropout=FLAGS.dropout
        ),
        attention_dropout=FLAGS.dropout,
        max_relative_positions=0,
    )
    decoder = TransformerDecoder(
        num_layers=FLAGS.num_layers,
        d_model=FLAGS.d_model,
        heads=FLAGS.n_heads,
        d_ff=FLAGS.d_ff,
        copy_attn=False,
        self_attn_type='scaled-dot',
        dropout=FLAGS.dropout,
        embeddings=Embeddings(
            word_vec_size=FLAGS.d_model,
            word_vocab_size=FLAGS.output_dim,
            word_padding_idx=FLAGS.trg_pad_idx,
            position_encoding=True,
            dropout=FLAGS.dropout
        ),
        aan_useffn=False,
        alignment_heads=0,
        alignment_layer=0,
        full_context_alignment=False,
        attention_dropout=FLAGS.dropout,
        max_relative_positions=0,
    )
    
    return NMTModel(encoder, decoder)

def get_tutorial_transformer():
    transfomer_encoder = Encoder(
            input_dim=FLAGS.input_dim, 
            hid_dim=FLAGS.d_model, 
            n_layers=FLAGS.num_layers, 
            n_heads=FLAGS.n_heads, 
            pf_dim=FLAGS.d_ff, 
            dropout=FLAGS.dropout,
            max_length=100)
    transfomer_decoder = Decoder(
        output_dim=FLAGS.output_dim, 
        hid_dim=FLAGS.d_model, 
        n_layers=FLAGS.num_layers, 
        n_heads=FLAGS.n_heads, 
        pf_dim=FLAGS.d_ff, 
        dropout=FLAGS.dropout,
        max_length=100)

    return Seq2Seq(
        encoder=transfomer_encoder, 
        decoder=transfomer_decoder,
        src_pad_idx=FLAGS.src_pad_idx, 
        trg_pad_idx=FLAGS.trg_pad_idx,
    )


class SignTransformer(pl.LightningModule):
    def __init__(self, data_dir, lr, weight_decay, model_name='onmt'):
        super().__init__()
        
        self.data_dir = data_dir
        self.weight_decay = weight_decay
        self.lr = lr
        self.model_name = model_name

        self.example_input_array = None

        if model_name == 'onmt':
            self.model = get_onmt_transformer()
            self.generator = nn.Sequential(
                nn.Linear(FLAGS.d_model, FLAGS.output_dim),
                Cast(torch.float32),
                nn.LogSoftmax(dim=-1)
            )
        elif model_name == 'tutorial':
            self.model = get_tutorial_transformer()
        self.model.apply(self.initialize_weights)
        
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=FLAGS.trg_pad_idx)
        
    def initialize_weights(self, m):
        if hasattr(m, 'weight') and m.weight.dim() > 1:
            nn.init.xavier_uniform_(m.weight.data)
    
    def setup(self, stage=None):
        def _tokenize(_string):
            return _string.split(' ')
        
        self.GLS = Field(
            tokenize=_tokenize,
            init_token='<sos>',
            eos_token='<eos>',
            batch_first=True,
            include_lengths=True,
        )
        self.TXT = Field(
            tokenize=_tokenize,
            init_token='<sos>',
            eos_token='<eos>',
            batch_first=True,
            include_lengths=True,
        )
        
        def _prepare_ds(mode):
            gls_path = os.path.join(self.data_dir, f'phoenix2014T.train.gloss')
            txt_path = os.path.join(self.data_dir, f'phoenix2014T.train.de')
            ds = SignTranslationData(gls_path, txt_path, (self.GLS, self.TXT))
            
            return ds
        
        self.train_data, self.val_data, self.test_data = map(_prepare_ds, ('train', 'dev', 'test'))
        
        # build vocab
        self.GLS.build_vocab(self.train_data, min_freq=1)
        self.TXT.build_vocab(self.train_data, min_freq=1)
        
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

    def forward(self, gls, txt, lengths=None):
        if self.model_name == 'onmt':
            gls = gls.unsqueeze(-1)
            txt = txt.unsqueeze(-1) # seq x bsz x 1
            output, _ = self.model(gls, txt, lengths)
            output = self.generator(output) #seq x bsz x output_dim
            output = output.transpose(1,0)
        elif self.model_name == 'tutorial':
            output, _ = self.model(gls, txt)
        
        return output

    def training_step(self, batch, batch_idx):
        txt, txt_lengths = batch.txt # bsz x seq
        gls, gls_lengths = batch.gls

        if self.model_name == 'onmt':
            # convert bsz x seq -> seq x bsz
            gls = gls.transpose(1,0)
            txt = txt.transpose(1,0)
            output = self.forward(gls, txt, gls_lengths)
            output_dim = output.shape[-1]
            txt = txt.transpose(1,0) # bsz x seq
           
        elif self.model_name == 'tutorial':
            output = self.forward(gls, txt[:, :-1])
            output_dim = output.shape[-1]
            
        pred = output.contiguous().view(-1, output_dim)    
        trg_txt = txt[:, 1:].contiguous().view(-1)
        
        loss = self.loss(pred, trg_txt)
        ppl = math.exp(loss)
        
        # get_blue
        _txt_hyp = self.get_txt_hyp(output)
        _txt_ref = self.get_txt_ref(txt)
        bleu = bleu_score(_txt_hyp, _txt_ref)
    
        # logging
        self.log('train_loss', loss, prog_bar=False, logger=True)
        self.log('train_ppl', ppl, prog_bar=False, logger=True)
        self.log('train_bleu', bleu, prog_bar=False, logger=True)
        self.log('lr', self.lr, prog_bar=True, logger=False)

        return loss  

    def validation_step(self, batch, batch_idx):
        txt, txt_lengths = batch.txt # bsz x seq
        gls, gls_lengths = batch.gls

        if self.model_name == 'onmt':
            # convert bsz x seq -> seq x bsz
            gls = gls.transpose(1,0)
            txt = txt.transpose(1,0)
            output = self.forward(gls, txt, gls_lengths)
            output_dim = output.shape[-1]
            txt = txt.transpose(1,0) # bsz x seq
           
        elif self.model_name == 'tutorial':
            output = self.forward(gls, txt[:, :-1])
            output_dim = output.shape[-1]
            
        pred = output.contiguous().view(-1, output_dim)    
        trg_txt = txt[:, 1:].contiguous().view(-1)
        
        loss = self.loss(pred, trg_txt)
        ppl = math.exp(loss)
        
        # get_blue
        _txt_hyp = self.get_txt_hyp(output)
        _txt_ref = self.get_txt_ref(txt)
        bleu = bleu_score(_txt_hyp, _txt_ref)
        
        # logging
        self.log('valid_loss', loss, prog_bar=False, logger=True)
        self.log('valid_ppl', ppl, prog_bar=False, logger=True)
        self.log('valid_bleu', bleu, prog_bar=False, logger=True) 

        return {
            'txt_hyp': _txt_hyp,
            'txt_ref': _txt_ref,
        }

    def validation_epoch_end(self, outputs):
        outputs = outputs[0]
        n_total = len(outputs['txt_ref'])
        idx = random.randrange(n_total)
        txt_hyp = outputs['txt_hyp'][idx]
        txt_ref = outputs['txt_ref'][idx]

        with open(os.path.join(FLAGS.save_log, 'gls2txt.output'), 'a') as f:
            f.write(f'txt hyp: {txt_hyp}\ntxt ref: {txt_ref}\n\n')

    def test_step(self, batch, batch_idx):
        import IPython; IPython.embed(); exit(1)

    def get_txt_hyp(self, logits):
        logits = logits.argmax(2)
        bsz, seq_len = logits.shape
        pred_txt = []
        for i in range(bsz):
            txt_hyp = []
            for j in range(seq_len):
                if logits[i][j] == self.TXT.vocab['<eos>']:
                    break
                txt_hyp.append(self.TXT.vocab.itos[logits[i][j]])
            pred_txt.append(txt_hyp)
        return [' '.join(token) for token in pred_txt]

    def get_txt_ref(self, txt):
        txt_ref_list = []
        for txt_indicies in txt:
            txt_ref = []
            for _txt_token in txt_indicies[1:]:
                if _txt_token == self.TXT.vocab.stoi['<eos>'] or _txt_token == self.TXT.vocab.stoi['<pad>']:
                    break
                txt_ref.append(self.TXT.vocab.itos[_txt_token])
            txt_ref_list.append(txt_ref)
        return [' '.join(token) for token in txt_ref_list]

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.lr,
            momentum=FLAGS.momentum,
            weight_decay=self.weight_decay,
        )
        scheduler = lr_scheduler.LambdaLR(
            optimizer,
            lambda epoch: 0.1 ** (epoch // 30)
        )

        return [optimizer], [scheduler] 


class SignTranslationData(Dataset):
    def __init__(self, gls_path: str, txt_path: str, fields: tuple):
        gls = self.read_data(gls_path)
        txt = self.read_data(txt_path)
        
        fields = [
            ('gls', fields[0]),
            ('txt', fields[1]),
        ]
        
        examples = self.get_examples(gls, txt, fields)
        super().__init__(examples, fields)

    def read_data(self, d_path):
        with open(d_path, 'r') as f:
            lines = f.readlines()
        return lines

    def get_examples(self, gls, txt, fields):
        examples = []
        for _gls, _txt in zip(gls, txt):
            examples.append(
                Example.fromlist(
                    [_gls, _txt],
                    fields,
                )
            )

        return examples

        
def main(_):
    sh.rm('-r', '-f', FLAGS.save_log)
    sh.mkdir(FLAGS.save_log)
    
    model = SignTransformer(FLAGS.data_path, FLAGS.lr, FLAGS.weight_decay, FLAGS.model_name)
    trainer = pl.Trainer(
        default_root_dir=FLAGS.save_log,
        gpus=(1 if torch.cuda.is_available() else 0),
        max_epochs=FLAGS.epochs,
        fast_dev_run=FLAGS.debug,
        gradient_clip_val=FLAGS.clip,
        logger=pl.loggers.TensorBoardLogger(
            save_dir=FLAGS.save_log,
            # name='log',
            log_graph=True,
            default_hp_metric=False
        )
        # auto_lr_find=True,
    )
    trainer.fit(model)
    # trainer.test()


if __name__ == "__main__":
    app.run(main)
