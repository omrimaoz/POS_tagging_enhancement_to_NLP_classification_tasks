import json
import sys
import time
from collections import Counter

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader

from LSTM.Dataset import ReviewsDataset
from sklearn.model_selection import train_test_split

from LSTM.Model import LSTM

dataset = sys.argv[1]
tagging = sys.argv[2]  # tagging = 'review' / 'upos' / 'xpos'
batch_size = 128
limit = -1
torch.manual_seed(0)


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
        val_loss, val_acc = validation_metrics(model, criterion, valid_loader)
        end = time.time()
        print('epoch: {epoch}, train_loss: {train_loss}, val_loss: {val_loss}, val_acc: {val_acc}, time: {time}s'.format(
            epoch=epoch, train_loss=np.round(sum_loss/total, 3), val_loss=np.round(val_loss, 3),
            val_acc=np.round(val_acc, 3), time=int(end-start)
        ))

def validation_metrics(model, criterion, valid_loader):
    model.eval()
    correct = 0
    total = 0
    sum_loss = 0.0
    for x, y, l in valid_loader:
        x = x.long()
        y = y.long()
        y_hat = model(x, l)
        loss = criterion(y_hat, y)
        pred = torch.max(y_hat, 1)[1]
        correct += (pred == y).float().sum()
        total += y.shape[0]
        sum_loss += loss.item()*y.shape[0]
    return sum_loss/total, correct/total

# loading the data
with open('../LSTM/Processed_' + tagging + '_' + dataset, 'r') as f:
    reviews = json.loads(f.read())
vocab_size = reviews['prop']['num_words']

# check how balanced the dataset is
print('How balanced the dataset is:')
print(Counter([review['sentiment'] for _, review in reviews['data'].items()]))

X = [(np.array(review['encode_sen']), review['len_sen']) for _, review in reviews['data'].items()][:limit]
y = [review['sentiment'] for _, review in reviews['data'].items()][:limit]
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)

train_ds = ReviewsDataset(X_train, y_train)
valid_ds = ReviewsDataset(X_valid, y_valid)
train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_dl = DataLoader(valid_ds, batch_size=batch_size)

model = LSTM(vocab_size, embedding_dim=30, hidden_dim=30, num_classes=reviews['prop']['num_classes'])

train_model(model, criterion=nn.CrossEntropyLoss(), train_loader=train_dl, valid_loader=val_dl, epochs=30, lr=0.001)
