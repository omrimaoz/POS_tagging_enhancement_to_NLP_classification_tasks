import time

import torch
from torch import nn

from Shared.F1_Score import get_binary_F1_score, get_multi_F1_score
from Transformer.Dataset import *
import torch.nn.functional as F
from Transformer.Batch import create_masks

from Transformer.Model import Transformer


def train_model(model, criterion, opt):
    print("training model...")
    for epoch in range(opt['epochs']):
        start = time.time()
        model.train()
        sum_loss = 0
        total = 0

        for i, batch in enumerate(opt['train']):
            src = batch.src.transpose(0, 1)
            trg = batch.trg.transpose(0, 1)
            trg_input = trg[:, :-1]
            src_mask, trg_mask = create_masks(src, trg_input, opt)
            preds = model(src, trg_input, src_mask, trg_mask)
            y = trg[:, 1:].contiguous().view(-1)
            opt['optimizer'].zero_grad()
            loss = F.cross_entropy(preds.view(-1, preds.size(-1)), y, ignore_index=opt['trg_pad'])
            loss.backward()
            opt['optimizer'].step()
            sum_loss += loss.item()
            total += y.shape[0]

        val_loss, val_acc, precision, recall, F1 = validate_model(model, criterion, opt)
        end = time.time()
        print('epoch: {epoch}, train_loss: {train_loss}, val_loss: {val_loss}, val_acc: {val_acc},'
              ' precision: {precision}, recall: {recall}, F1: {F1}, time: {time}s'.format(
            epoch=epoch, train_loss=np.round(sum_loss, 3), val_loss=np.round(val_loss, 3),
            val_acc=np.round(val_acc, 3), precision=np.round(precision, 3), recall=np.round(recall, 3),
            F1=np.round(F1, 3), time=int(end - start)
        ))

def validate_model(model, criterion, opt):
    model.eval()
    correct = 0
    total = 0
    sum_loss = 0.0
    pred = None
    actual = None
    get_F1_score = get_binary_F1_score if model.trg_encode - 4 == 2 else get_multi_F1_score
    for i, batch in enumerate(opt['valid']):
        src = batch.src.transpose(0, 1)
        trg = batch.trg.transpose(0, 1)
        trg_input = trg[:, :-1]
        src_mask, trg_mask = create_masks(src, trg_input, opt)
        y_hat = model(src, trg_input, src_mask, trg_mask)
        y = trg[:, 1:].contiguous().view(-1)
        actual = np.array(y) if actual is None else np.concatenate((actual, y))
        y_hat = y_hat.view(-1, y_hat.size(-1))
        loss = criterion(y_hat, y, ignore_index=opt['trg_pad'])

        y_hat = torch.max(y_hat, 1)[1]
        pred = np.array(y_hat) if pred is None else np.concatenate((pred, y_hat))
        correct += (y_hat == y).float().sum()
        total += y.shape[0]
        sum_loss += loss.item()
    accuracy, precision, recall, F1 = get_F1_score(actual, pred)
    return sum_loss, accuracy, precision, recall, F1


def main(dataset_name, tag_feature, limit):
    # limit must divide by batch-size
    limit = 1024 if limit == 1000 else limit
    limit = 4096 if limit == 5000 else limit
    limit = 14848 if limit == 15000 else limit

    if tag_feature == 'original':
        path = './Datasets/{dataset_name}_Dataset_15000_upos.json'.format(
            dataset_name=dataset_name)
    else:
        path = './Datasets/{dataset_name}_Dataset_15000_{tag_feature}.json'.format(
            dataset_name=dataset_name, tag_feature=tag_feature)
        tag_feature = 'upos'

    opt = {
        'data': path,
        'lang': 'en_core_web_sm',
        'epochs': 50,
        'd_model': 32,
        'n_layers': 2,
        'heads': 4,
        'dropout': 0.1,
        'batchsize': 256,
        'lr': 0.01,
        'train_valid_ratio': 0.75,
        'max_strlen': 512,
        'tagging': tag_feature,
        'limit': limit
    }
    torch.manual_seed(0)

    opt['train'], opt['valid'], SRC, TRG = create_dataset(opt)
    model = Transformer(len(SRC.vocab), len(TRG.vocab), opt['d_model'], opt['n_layers'], opt['heads'], opt['dropout'])
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    opt['optimizer'] = torch.optim.Adam(model.parameters(), lr=opt['lr'], betas=(0.9, 0.98), eps=1e-9)

    train_model(model, criterion=F.cross_entropy, opt=opt)
