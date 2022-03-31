import time

import numpy as np
from torch import nn
from Dataset import *
import torch.nn.functional as F
from Optim import CosineWithRestarts
from Batch import create_masks
import dill as pickle

from Transformer.Model import Transformer


def train_model(model, criterion, opt):
    print("training model...")
    # if opt.['checkpoint'] > 0:
    #     cptime = time.time()

    for epoch in range(opt['epochs']):
        start = time.time()
        model.train()
        sum_loss = 0
        total = 0
        if opt['floyd'] is False:
            print("   %dm: epoch %d [%s]  %d%%  loss = %s" % \
                  ((time.time() - start) // 60, epoch + 1, "".join(' ' * 20), 0, '...'), end='\r')

        # if opt['checkpoint'] > 0:
        #     torch.save(model.state_dict(), 'weights/model_weights')

        for i, batch in enumerate(opt['train']):
            # batches = [batch for batch in opt['train']]
            src = batch.src.transpose(0, 1)
            trg = batch.trg.transpose(0, 1)
            trg_input = trg[:, 1:-1]
            src_mask, trg_mask = create_masks(src, trg_input, opt)
            delete_rows = 1 - ((trg_input == (model.trg_encode - 1)) * 1).view(-1)
            preds = model(src, trg_input, src_mask, trg_mask)
            preds = (preds.view(-1, preds.size(-1)).T * delete_rows).T
            preds_max = torch.max(preds, 1)[1]
            y = trg[:, 1:-1].contiguous().view(-1) - 4
            y = y * delete_rows
            loss = criterion(preds, y) #, ignore_index=opt['trg_pad'])
            opt['optimizer'].zero_grad()
            loss.backward()
            opt['optimizer'].step()
            if opt['SGDR'] == True:
                opt['sched'].step()

            sum_loss += loss.item()
            total += trg_input.shape[0]

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
        trg_input = trg[:, 1:-1]
        src_mask, trg_mask = create_masks(src, trg_input, opt)
        delete_rows = 1 - ((trg_input == (model.trg_encode-1)) * 1).view(-1)
        preds = model(src, trg_input, src_mask, trg_mask)
        preds = (preds.view(-1, preds.size(-1)).T * delete_rows).T

        y = trg[:, 1:-1].contiguous().view(-1) - 4
        y = y * delete_rows
        loss = criterion(preds, y) #, ignore_index=opt['trg_pad'])

        preds = torch.max(preds, 1)[1]
        correct += (preds == y).float().sum().item()
        total += trg_input.shape[0]
        sum_loss += loss.item()
    return sum_loss, correct / total

            # if (i + 1) % opt['printevery'] == 0:
            #     p = int(100 * (i + 1) / opt['train_len'])
            #     avg_loss = total_loss / opt['printevery']
            #     if opt['floyd'] is False:
            #         print("   %dm: epoch %d [%s%s]  %d%%  loss = %.3f" % \
            #               ((time.time() - start) // 60, epoch + 1, "".join('#' * (p // 5)),
            #                "".join(' ' * (20 - (p // 5))), p, avg_loss), end='\r')
            #     else:
            #         print("   %dm: epoch %d [%s%s]  %d%%  loss = %.3f" % \
            #               ((time.time() - start) // 60, epoch + 1, "".join('#' * (p // 5)),
            #                "".join(' ' * (20 - (p // 5))), p, avg_loss))
            #     total_loss = 0

            # if opt.['checkpoint'] > 0 and ((time.time() - cptime) // 60) // opt.['checkpoint'] >= 1:
            #     torch.save(model.state_dict(), 'weights/model_weights')
            #     cptime = time.time()

        # print("%dm: epoch %d [%s%s]  %d%%  loss = %.3f\nepoch %d complete, loss = %.03f" % \
        #       ((time.time() - start) // 60, epoch + 1, "".join('#' * (100 // 5)), "".join(' ' * (20 - (100 // 5))), 100,
        #        avg_loss, epoch + 1, avg_loss))


def yesno(response):
    while True:
        if response != 'y' and response != 'n':
            response = input('command not recognised, enter y or n : ')
        else:
            return response


def promptNextAction(model, opt, SRC, TRG):
    saved_once = 1 if opt.load_weights is not None or opt.checkpoint > 0 else 0

    if opt.load_weights is not None:
        dst = opt.load_weights
    if opt.checkpoint > 0:
        dst = 'weights'

    while True:
        save = yesno(input('training complete, save results? [y/n] : '))
        if save == 'y':
            while True:
                if saved_once != 0:
                    res = yesno("save to same folder? [y/n] : ")
                    if res == 'y':
                        break
                dst = input('enter folder name to create for weights (no spaces) : ')
                if ' ' in dst or len(dst) < 1 or len(dst) > 30:
                    dst = input(
                        "name must not contain spaces and be between 1 and 30 characters length, enter again : ")
                else:
                    try:
                        os.mkdir(dst)
                    except:
                        res = yesno(input(dst + " already exists, use anyway? [y/n] : "))
                        if res == 'n':
                            continue
                    break

            print("saving weights to " + dst + "/...")
            torch.save(model.state_dict(), f'{dst}/model_weights')
            if saved_once == 0:
                pickle.dump(SRC, open(f'{dst}/SRC.pkl', 'wb'))
                pickle.dump(TRG, open(f'{dst}/TRG.pkl', 'wb'))
                saved_once = 1

            print("weights and field pickles saved to " + dst)

        res = yesno(input("train for more epochs? [y/n] : "))
        if res == 'y':
            while True:
                epochs = input("type number of epochs to train for : ")
                try:
                    epochs = int(epochs)
                except:
                    print("input not a number")
                    continue
                if epochs < 1:
                    print("epochs must be at least 1")
                    continue
                else:
                    break
            opt.epochs = epochs
            train_model(model, opt)
        else:
            print("exiting program...")
            break

    # for asking about further training use while true loop, and return

opt = {
    'data': '../Datasets/IMDB_Dataset_5000.json',
    'lang': 'en_core_web_sm',
    'no_cuda': True,
    'SGDR': False,
    'epochs': 50,
    'd_model': 8,
    'n_layers': 1,
    'heads': 4,
    'dropout': 0.1,
    'batchsize': 256,
    'printevery': 100,
    'lr': 0.0001,
    'train_valid_ratio': 0.8,
    'load_weights': None,
    'create_valset': True,
    'max_strlen': 512,
    'floyd': True,
    'checkpoint': 0,
    'tagging': 'upos'
}
torch.manual_seed(0)

opt['device'] = 0 if opt['no_cuda'] is False else -1
if opt['device'] == 0:
    assert torch.cuda.is_available()

opt['train'], opt['valid'], SRC, TRG = create_dataset(opt)
model = Transformer(len(SRC.vocab), len(TRG.vocab), opt['d_model'], opt['n_layers'], opt['heads'], opt['dropout'])
for p in model.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

opt['optimizer'] = torch.optim.Adam(model.parameters(), lr=opt['lr'], betas=(0.9, 0.98), eps=1e-9)
if opt['SGDR'] == True:
    opt['sched'] = CosineWithRestarts(opt['optimizer'], T_max=opt['train_len'])

# if opt['checkpoint'] > 0:
#     print(
#         "model weights will be saved every %d minutes and at end of epoch to directory weights/" % (opt['checkpoint']))

# if opt['load_weights'] is not None and opt['floyd'] is not None:
#     os.mkdir('weights')
#     pickle.dump(SRC, open('weights/SRC.pkl', 'wb'))
#     pickle.dump(TRG, open('weights/TRG.pkl', 'wb'))

train_model(model, criterion=F.cross_entropy, opt=opt)

if opt['floyd'] is False:
    promptNextAction(model, opt, SRC, TRG)
