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
import torch.optim.lr_scheduler as lr_scheduler
import torchtext
from absl import app, flags, logging
from torchtext.data import (BucketIterator, Dataset, Example, Field,
                            TabularDataset)

from metric import bleu as bleu_score
from onmt.decoders.transformer import TransformerDecoder
from onmt.encoders.transformer import TransformerEncoder
from onmt.models.model import NMTModel
from onmt.modules import Embeddings
from onmt.modules.util_class import Cast

from model.transformer import Decoder, Encoder, Seq2Seq

# sh.rm('-r', '-f', 'tutorial_log')
# sh.mkdir('tutorial_log')

flags.DEFINE_integer('input_dim', 1814, '')
flags.DEFINE_integer('output_dim', 734, '')
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
flags.DEFINE_integer('epochs', 1000, '')
flags.DEFINE_float('lr', 0.01, '')
flags.DEFINE_float('clip', .5, '')
flags.DEFINE_float('momentum', .9, '')
flags.DEFINE_float('weight_decay', 1e-4, '')

flags.DEFINE_string('data_path', './annotation', '')
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

        self.TXT = Field(
            tokenize=_tokenize,
            init_token='<sos>',
            eos_token='<eos>',
            batch_first=True,
            include_lengths=True,
        )
        self.GLS = Field(
            tokenize=_tokenize,
            init_token='<sos>',
            eos_token='<eos>',
            batch_first=True,
            include_lengths=True,
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

    def forward(self, txt, gls, lengths=None):
        if self.model_name == 'onmt':
            txt = txt.unsqueeze(-1) # seq x bsz x 1
            gls = gls.unsqueeze(-1)
            output, _ = self.model(txt, gls, lengths)
            output = self.generator(output) #seq x bsz x output_dim
            output = output.transpose(1,0)
        elif self.model_name == 'tutorial':
            output, _ = self.model(txt, gls)
        
        return output

    def training_step(self, batch, batch_idx):
        txt, txt_lengths = batch.txt # bsz x seq
        gls, gls_lengths = batch.gls

        if self.model_name == 'onmt':
            # convert bsz x seq -> seq x bsz
            txt = txt.transpose(1,0)
            gls = gls.transpose(1,0)
            output = self.forward(txt, gls, txt_lengths)
            output_dim = output.shape[-1]
            gls = gls.transpose(1,0) # bsz x seq
           
        elif self.model_name == 'tutorial':
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
    
        # logging
        self.log('train_loss', loss, prog_bar=False, logger=True)
        self.log('train_ppl', ppl, prog_bar=False, logger=True)
        self.log('train_bleu', bleu, prog_bar=False, logger=True) 

        return loss  

    def validation_step(self, batch, batch_idx):
        txt, txt_lengths = batch.txt # bsz x seq
        gls, gls_lengths = batch.gls

        if self.model_name == 'onmt':
            # convert bsz x seq -> seq x bsz
            txt = txt.transpose(1,0)
            gls = gls.transpose(1,0)
            output = self.forward(txt, gls, txt_lengths)
            output_dim = output.shape[-1]
            gls = gls.transpose(1,0) # bsz x seq
           
        elif self.model_name == 'tutorial':
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
        
        # logging
        self.log('valid_loss', loss, prog_bar=False, logger=True)
        self.log('valid_ppl', ppl, prog_bar=False, logger=True)
        self.log('valid_bleu', bleu, prog_bar=False, logger=True) 

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
    model = SignTransformer(FLAGS.data_path, FLAGS.lr, FLAGS.weight_decay, FLAGS.model_name)
    
    trainer = pl.Trainer(
        default_root_dir='tutorial_log',
        gpus=(1 if torch.cuda.is_available() else 0),
        max_epochs=FLAGS.epochs,
        fast_dev_run=FLAGS.debug,
        # gradient_clip_val=FLAGS.clip,
        # auto_lr_find=True,
    )
    trainer.fit(model)
    # trainer.test()


if __name__ == "__main__":
    app.run(main)
