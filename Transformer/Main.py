import sys
import time

import torch
from torch import nn
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

            # src = batch.src.transpose(0, 1)
            # trg = batch.trg.transpose(0, 1)
            # trg_input = trg[:, 1:-1] - 4
            # src_mask, trg_mask = create_masks(src, trg_input, opt)
            # delete_rows = 1 - ((trg_input == (model.trg_encode - 1)) * 1).view(-1)
            # preds = model(src, trg_input, src_mask, trg_mask)
            # preds = (preds.view(-1, preds.size(-1)).T * delete_rows).T
            # preds_max = torch.max(preds, 1)[1]
            # y = trg[:, 1:-1].contiguous().view(-1) - 4
            # y = y * delete_rows
            # try:
            #     loss = criterion(preds, y) #, ignore_index=opt['trg_pad'])
            # except:
            #     print(1)
            # opt['optimizer'].zero_grad()
            # loss.backward()
            # opt['optimizer'].step()
            # if opt['SGDR'] == True:
            #     opt['sched'].step()
            #
            # sum_loss += loss.item()
            total += y.shape[0]

        val_loss, val_acc = validate_model(model, criterion, opt)
        end = time.time()
        print(
            'epoch: {epoch}, train_loss: {train_loss}, val_loss: {val_loss}, val_acc: {val_acc}, time: {time}s'.format(
                epoch=epoch, train_loss=np.round(sum_loss, 3), val_loss=np.round(val_loss, 3),
                val_acc=np.round(val_acc, 3), time=int(end - start)
            ))

def validate_model(model, criterion, opt):
    model.eval()
    correct = 0
    total = 0
    sum_loss = 0.0
    for i, batch in enumerate(opt['valid']):

        src = batch.src.transpose(0, 1)
        trg = batch.trg.transpose(0, 1)
        trg_input = trg[:, :-1]
        src_mask, trg_mask = create_masks(src, trg_input, opt)
        preds = model(src, trg_input, src_mask, trg_mask)
        y = trg[:, 1:].contiguous().view(-1)
        preds = preds.view(-1, preds.size(-1))
        loss = F.cross_entropy(preds, y, ignore_index=opt['trg_pad'])

        # src = batch.src.transpose(0, 1)
        # trg = batch.trg.transpose(0, 1)
        # trg_input = trg[:, 1:-1] - 4
        # src_mask, trg_mask = create_masks(src, trg_input, opt)
        # delete_rows = 1 - ((trg_input == (model.trg_encode-1)) * 1).view(-1)
        # preds = model(src, trg_input, src_mask, trg_mask)
        # preds = (preds.view(-1, preds.size(-1)).T * delete_rows).T
        #
        # y = trg[:, 1:-1].contiguous().view(-1) - 4
        # y = y * delete_rows
        # loss = criterion(preds, y) #, ignore_index=opt['trg_pad'])

        preds = torch.max(preds, 1)[1]
        correct += (preds == y).float().sum().item()
        total += y.shape[0]
        sum_loss += loss.item()
    return sum_loss, correct / total


def main(dataset_name, tag_feature, limit, dataset_size):
    # limit must divide by batch-size
    limit = 1024 if limit == 1000 else limit
    limit = 4096 if limit == 5000 else limit
    limit = 14848 if limit == 15000 else limit

    if tag_feature == 'original':
        path = '../Datasets/{dataset_name}_Dataset_{dataset_size}_upos.json'.format(
            dataset_name=dataset_name, dataset_size=dataset_size)
    else:
        path = '../Datasets/{dataset_name}_Dataset_{dataset_size}_{tag_feature}.json'.format(
            dataset_name=dataset_name, dataset_size=dataset_size, tag_feature=tag_feature)
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
