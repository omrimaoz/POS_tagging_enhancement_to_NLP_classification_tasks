import json
import random

import numpy as np
import pandas as pd
from torchtext.legacy import data

from Shared.Common_Functions import remove_punctuation
from Transformer.Tokenize import tokenize
import os


def create_dataset(opt):
    opt['data'] = json.loads(open(opt['data']).read())
    opt['src_data'] = [phrase[opt['tagging']] for _, phrase in opt['data'].items()]
    opt['trg_data'] = [phrase['class'] for _, phrase in opt['data'].items()]

    print("loading spacy tokenizers...")
    t_src = tokenize(opt['lang'])
    t_trg = tokenize(opt['lang'])

    TRG = data.Field(lower=True, tokenize=t_trg.tokenizer, init_token='<sos>', eos_token='<eos>')
    SRC = data.Field(lower=True, tokenize=t_src.tokenizer)

    print("creating dataset and iterator... ")

    raw_data = {'src': [remove_punctuation(line) for line in opt['src_data']], 'trg': [line for line in opt['trg_data']]}
    df = pd.DataFrame(raw_data, columns=["src", "trg"])

    mask = (df['src'].str.count(' ') < opt['max_strlen'])# & (df['trg'].str.count(' ') < opt['max_strlen'])
    df = df.loc[mask][:opt['limit']]

    rows = [i for i in range(df.shape[0])]
    random.shuffle(rows)
    train_rows = rows[:int(len(rows) * opt['train_valid_ratio'])]
    valid_rows = rows[int(len(rows) * opt['train_valid_ratio']):]

    df[np.isin(np.arange(len(rows)), train_rows)].to_csv("translate_transformer_train_temp.csv", index=False)
    df[np.isin(np.arange(len(rows)), valid_rows)].to_csv("translate_transformer_valid_temp.csv", index=False)

    data_fields = [('src', SRC), ('trg', TRG)]
    ds_train = data.TabularDataset('./translate_transformer_train_temp.csv', format='csv', fields=data_fields)
    ds_valid = data.TabularDataset('./translate_transformer_valid_temp.csv', format='csv', fields=data_fields)

    ds_train.examples.pop(0)
    ds_valid.examples.pop(0)
    # dl_train = torch.utils.data.DataLoader(
    #     ds_train, batch_size=opt['batchsize'], shuffle=True)
    # dl_valid = torch.utils.data.DataLoader(
    #     ds_valid, batch_size=opt['batchsize'], shuffle=False)

    train_iter = data.Iterator(ds_train, batch_size=opt['batchsize'],
                            repeat=False, train=True, shuffle=True)
    valid_iter = data.Iterator(ds_valid, batch_size=opt['batchsize'],
                            repeat=False, train=True, shuffle=True)

    # train_iter = MyIterator(ds_train, batch_size=opt['batchsize'], device=opt['device'],
    #                         repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
    #                         batch_size_fn=batch_size_fn, train=True, shuffle=True)
    # valid_iter = MyIterator(ds_valid, batch_size=opt['batchsize'], device=opt['device'],
    #                         repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
    #                         batch_size_fn=batch_size_fn, train=False, shuffle=False)

    os.remove('translate_transformer_train_temp.csv')
    os.remove('translate_transformer_valid_temp.csv')

    SRC.build_vocab(ds_train)
    TRG.build_vocab(ds_valid)

    opt['src_pad'] = SRC.vocab.stoi['<pad>']
    opt['trg_pad'] = TRG.vocab.stoi['<pad>']

    opt['train_len'] = len([_ for _ in train_iter])
    opt['valid_len'] = len([_ for _ in valid_iter])

    return train_iter, valid_iter, SRC, TRG
