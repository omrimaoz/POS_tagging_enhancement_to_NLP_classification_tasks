import json
import sys
import time
from collections import Counter

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from DAN.Dataset import PhrasesDataset
from DAN.Encode_Dataset import encode_dataset
from DAN.Model import DAN

dataset_name = sys.argv[1]
tags = sys.argv[2]  # tags = 'original' / 'upos' / 'xpos'
epochs = 50
limit = 5000
# torch.manual_seed(2)


def train_model(model, criterion, train_loader, valid_loader, epochs=10, lr=0.01):
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=lr)
    for epoch in range(epochs):
        start = time.time()
        model.train()
        sum_loss = 0.0
        total = 0
        for batch, pack in enumerate(train_loader):
            x, y, l = pack
            x = x.long()
            y = y.long()
            y_pred = model(x, l)

            loss = criterion(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sum_loss += loss.item()*y.shape[0]
            total += y.shape[0]
            # if batch % 10:
            #     print({'epoch': epoch, 'batch': batch})
        val_loss, val_acc, precision, recall, F1 = validate_model(
            model, criterion, valid_loader)
        end = time.time()
        print('epoch: {epoch}, train_loss: {train_loss}, val_loss: {val_loss}, val_acc: {val_acc},'
              ' precision: {precision}, recall: {recall}, F1: {F1}, time: {time}s'.format(
                  epoch=epoch, train_loss=np.round(sum_loss/total, 3), val_loss=np.round(val_loss, 3),
                  val_acc=np.round(val_acc, 3), precision=np.round(precision, 3), recall=np.round(recall, 3),
                  F1=np.round(F1, 3), time=int(end-start)
              ))


def validate_model(model, criterion, valid_loader):
    model.eval()
    correct = 0
    total = 0
    sum_loss = 0.0
    pred = None
    actual = None
    get_F1_score = get_binary_F1_score if model.num_classes == 2 else get_multi_F1_score
    for x, y, l in valid_loader:
        x = x.long()
        y = y.long()
        actual = np.array(y) if actual is None else np.concatenate((actual, y))
        y_hat = model(x, l)
        loss = criterion(y_hat, y)
        pred = np.array(torch.max(y_hat, 1)[1]) if pred is None else np.concatenate(
            (pred, torch.max(y_hat, 1)[1]))
        correct += (torch.max(y_hat, 1)[1] == y).float().sum()
        total += y.shape[0]
        sum_loss += loss.item()*y.shape[0]
    accuracy, precision, recall, F1 = get_F1_score(actual, pred)
    return sum_loss/total, accuracy, precision, recall, F1


# loading the data
with open("../Datasets/" + dataset_name, 'r') as f:
    dataset = json.loads(f.read())

# encoding the data
phrases = encode_dataset(dataset, tags)

for key, item in phrases['data'].copy().items():
    if item['class'] > 3:
        phrases['data'].pop(key)


vocab_size = phrases['prop']['num_words']
batch_size = min(phrases['prop']['num_phrases'],
                 limit) // 10  # len(phrases['data']) // 10

X = [(np.array(phrase['encode_sen']), phrase['len_sen'])
     for _, phrase in phrases['data'].items()][:limit]
y = [phrase['class'] for _, phrase in phrases['data'].items()][:limit]

# check how balanced the dataset is
print('How balanced the dataset is:')
print(Counter(y))

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)

train_ds = PhrasesDataset(X_train, y_train)
valid_ds = PhrasesDataset(X_valid, y_valid)
train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_dl = DataLoader(valid_ds, batch_size=batch_size)


model = LSTM(vocab_size, embedding_dim=300, hidden_dim=128,
             num_classes=phrases['prop']['num_classes'])
# model = LSTM3(vocab_size, embedding_dim=vocab_size//2)

train_model(model, criterion=nn.CrossEntropyLoss(),
            train_loader=train_dl, valid_loader=val_dl, epochs=epochs, lr=0.01)
# train_model(model, criterion=nn.BCELoss(), train_loader=train_dl, valid_loader=val_dl, epochs=epochs, lr=0.001)
# test_loss, test_acc = validation_metrics(model, criterion=nn.CrossEntropyLoss(), valid_loader=test_dl)
# print('Test results:\ntest_loss: {test_loss}, test_acc: {test_acc}'.format(
#     test_loss=np.round(test_loss, 3), test_acc=np.round(float(test_acc), 3)))
