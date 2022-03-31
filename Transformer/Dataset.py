import json
import random
import torch
import pandas as pd
from torchtext.legacy import data

from Tokenize import tokenize
from Transformer_try1.helper.Batch import MyIterator, batch_size_fn
import os


# def read_data(opt):
#     if opt.src_data is not None:
#         try:
#             opt.src_data = open(opt.src_data).read().strip().split('\n')
#         except:
#             print("error: '" + opt.src_data + "' file not found")
#             quit()
#
#     if opt.trg_data is not None:
#         try:
#             opt.trg_data = open(opt.trg_data).read().strip().split('\n')
#         except:
#             print("error: '" + opt.trg_data + "' file not found")
#             quit()
#
#
# def create_fields(opt):
#     spacy_langs = ['en_core_web_sm', 'fr_core_news_sm', 'de', 'es', 'pt', 'it', 'nl']
#     if opt.src_lang not in spacy_langs:
#         print('invalid src language: ' + opt.src_lang + 'supported languages : ' + spacy_langs)
#     if opt.trg_lang not in spacy_langs:
#         print('invalid trg language: ' + opt.trg_lang + 'supported languages : ' + spacy_langs)
#
#     print("loading spacy tokenizers...")
#
#     t_src = tokenize(opt.src_lang)
#     t_trg = tokenize(opt.trg_lang)
#
#     TRG = data.Field(lower=True, tokenize=t_trg.tokenizer, init_token='<sos>', eos_token='<eos>')
#     SRC = data.Field(lower=True, tokenize=t_src.tokenizer)
#
#     if opt.load_weights is not None:
#         try:
#             print("loading presaved fields...")
#             SRC = pickle.load(open(f'{opt.load_weights}/SRC.pkl', 'rb'))
#             TRG = pickle.load(open(f'{opt.load_weights}/TRG.pkl', 'rb'))
#         except:
#             print("error opening SRC.pkl and TXT.pkl field files, please ensure they are in " + opt.load_weights + "/")
#             quit()
#
#     return (SRC, TRG)


def create_dataset(opt):
    # opt['src_data'] = open(opt['src_data']).read().strip().split('\n')
    # opt['trg_data'] = open(opt['trg_data']).read().strip().split('\n')
    opt['data'] = json.loads(open(opt['data']).read())
    opt['src_data'] = [phrase[opt['tagging']] for _, phrase in opt['data'].items()]
    opt['trg_data'] = [phrase['class'] for _, phrase in opt['data'].items()]

    print("loading spacy tokenizers...")
    t_src = tokenize(opt['lang'])
    t_trg = tokenize(opt['lang'])

    TRG = data.Field(lower=True, tokenize=t_trg.tokenizer, init_token='<sos>', eos_token='<eos>')
    SRC = data.Field(lower=True, tokenize=t_src.tokenizer)

    print("creating dataset and iterator... ")

    raw_data = {'src': [line for line in opt['src_data']], 'trg': [line for line in opt['trg_data']]}
    df = pd.DataFrame(raw_data, columns=["src", "trg"])

    mask = (df['src'].str.count(' ') < opt['max_strlen'])# & (df['trg'].str.count(' ') < opt['max_strlen'])
    df = df.loc[mask]

    rows = [i for i in range(df.shape[0])]
    random.shuffle(rows)
    train_rows = rows[:int(len(rows) * opt['train_valid_ratio'])]
    valid_rows = rows[int(len(rows) * opt['train_valid_ratio']):]

    df[df.index.isin(train_rows)].to_csv("translate_transformer_train_temp.csv", index=False)
    df[df.index.isin(valid_rows)].to_csv("translate_transformer_valid_temp.csv", index=False)

    data_fields = [('src', SRC), ('trg', TRG)]
    ds_train = data.TabularDataset('./translate_transformer_train_temp.csv', format='csv', fields=data_fields)
    ds_valid = data.TabularDataset('./translate_transformer_valid_temp.csv', format='csv', fields=data_fields)

    # dl_train = torch.utils.data.DataLoader(
    #     ds_train, batch_size=opt['batchsize'], shuffle=True)
    # dl_valid = torch.utils.data.DataLoader(
    #     ds_valid, batch_size=opt['batchsize'], shuffle=False)

    train_iter = data.Iterator(ds_train, batch_size=opt['batchsize'],
                            repeat=False,train=True, shuffle=True)
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


    if opt['load_weights'] is None:
        SRC.build_vocab(ds_train)
        TRG.build_vocab(ds_valid)
        # if opt.checkpoint > 0:
        #     try:
        #         os.mkdir("weights")
        #     except:
        #         print("weights folder already exists, run program with -load_weights weights to load them")
        #         quit()
        #     pickle.dump(SRC, open('weights/SRC.pkl', 'wb'))
        #     pickle.dump(TRG, open('weights/TRG.pkl', 'wb'))

    opt['src_pad'] = SRC.vocab.stoi['<pad>']
    opt['trg_pad'] = TRG.vocab.stoi['<pad>']

    opt['train_len'] = len([_ for _ in train_iter])
    opt['valid_len'] = len([_ for _ in valid_iter])

    return train_iter, valid_iter, SRC, TRG
