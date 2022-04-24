import json
import time
from collections import Counter

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from BiLSTM.Dataset import PhrasesDataset
from sklearn.model_selection import train_test_split
from BiLSTM.Encode_Dataset import encode_dataset
from BiLSTM.Model import BiLSTM
from Shared.F1_Score import get_binary_F1_score, get_multi_F1_score


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

        val_loss, val_acc, precision, recall, F1 = validate_model(model, criterion, valid_loader)
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
        y_hat = torch.max(y_hat, 1)[1]
        pred = np.array(y_hat) if pred is None else np.concatenate((pred, y_hat))
        correct += (y_hat == y).float().sum()
        total += y.shape[0]
        sum_loss += loss.item()*y.shape[0]
    accuracy, precision, recall, F1 = get_F1_score(actual, pred)
    return sum_loss/total, accuracy, precision, recall, F1


def main(dataset_name, tag_feature, limit):
    # loading the data
    if tag_feature == 'original':
        path = './Datasets/{dataset_name}_Dataset_15000_upos.json'.format(
            dataset_name=dataset_name)
    else:
        path = './Datasets/{dataset_name}_Dataset_15000_{tag_feature}.json'.format(
            dataset_name=dataset_name, tag_feature=tag_feature)
        tag_feature = 'upos'

    with open(path, 'r') as f:
        dataset = json.loads(f.read())

    # encoding the data
    phrases = encode_dataset(dataset, tag_feature)

    vocab_size = phrases['prop']['num_words']
    batch_size = min(phrases['prop']['num_phrases'], limit) // 10
    print('Number of phrases: {}'.format(min(phrases['prop']['num_phrases'], limit)))

    X = [(np.array(phrase['encode_sen']), phrase['len_sen']) for _, phrase in phrases['data'].items()][:limit]
    y = [phrase['class'] for _, phrase in phrases['data'].items()][:limit]

    # check how balanced the dataset is
    print('How balanced the dataset is:')
    print(Counter(y))

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)

    train_ds = PhrasesDataset(X_train, y_train)
    valid_ds = PhrasesDataset(X_valid, y_valid)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(valid_ds, batch_size=batch_size)

    model = BiLSTM(vocab_size, embedding_dim=300, hidden_dim=128, num_classes=phrases['prop']['num_classes'])

    train_model(model, criterion=nn.CrossEntropyLoss(), train_loader=train_dl, valid_loader=val_dl, epochs=50, lr=0.005)
