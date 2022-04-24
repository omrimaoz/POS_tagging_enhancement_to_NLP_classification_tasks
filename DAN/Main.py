import nltk
import gdown
import os
import numpy as np

from DAN.PrepareDatasets import load_json
from DAN.Model import DAN
from nltk.tokenize import word_tokenize
import json
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from dataclasses import dataclass
import torch

from transformers import Trainer
from transformers import TrainingArguments

import matplotlib.pyplot as plt


def tokenize_function(example, vocab):
    sentences = [x.lower() for x in example['text']]
    tokenized_sentences = [word_tokenize(x) for x in sentences]
    tokenized_idx = [[vocab[word] if word in vocab else vocab["unk"] for word in x] for x in tokenized_sentences]
    final_tokenized_idx = tokenized_idx

    return {"labels": example['label'], 'input_ids': final_tokenized_idx}


def pad_sequence_to_length(
        sequence,
        desired_length: int,
        default_value=lambda: 0,
        padding_on_right: bool = True,
):
    sequence = list(sequence)
    if padding_on_right:
        padded_sequence = sequence[:desired_length]
    else:
        padded_sequence = sequence[-desired_length:]
    pad_length = desired_length - len(padded_sequence)

    values_to_pad = [default_value()] * pad_length
    if padding_on_right:
        padded_sequence = padded_sequence + values_to_pad
    else:
        padded_sequence = values_to_pad + padded_sequence
    return padded_sequence


def compute_metrics(eval_pred):
    logits = eval_pred.predictions
    labels = eval_pred.label_ids
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='macro')
    accuracy = accuracy_score(labels, predictions)

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy
    }


@dataclass
class DataCollatorWithPadding:

    def __call__(self, features):
        features_dict = {}
        if "labels" in features[0]:
            features_dict["labels"] = torch.tensor([x.pop("labels") for x in features]).long()

        input_ids = [x.pop("input_ids") for x in features]
        max_len = max(len(x) for x in input_ids)
        masks = [[1] * len(x) for x in input_ids]

        features_dict["input_ids"] = torch.tensor([pad_sequence_to_length(x, max_len) for x in input_ids]).long()
        features_dict["attention_masks"] = torch.tensor([pad_sequence_to_length(x, max_len) for x in masks]).long()

        return features_dict

def get_XY(trainer, metric):
    hist = trainer.state.log_history

    epoch_eval = [(x["epoch"], x.get(f"eval_{metric}", {None: None}).get(metric)) for x in hist if
                  int(x["epoch"]) == x["epoch"]]
    epoch_eval = [x for x in epoch_eval if x[1] is not None]

    epoch_eval = sorted(epoch_eval, key=lambda x: x[0])

    X = [x for x, y in epoch_eval]

    Y = [y * 100 for x, y in epoch_eval]

    return X, Y


def mkdir_p(mypath):
    '''Creates a directory. equivalent to using mkdir -p on the command line'''

    from errno import EEXIST
    from os import makedirs, path

    try:
        makedirs(mypath)
    except OSError as exc:  # Python >2.5
        if exc.errno == EEXIST and path.isdir(mypath):
            pass
        else:
            raise


def plot_trainer(trainer, title, metric, file_name, mode):
    colors = {
        'accuracy': 'lightblue',
        'f1': 'orange',
        'recall': 'green',
        'precision': 'purple'
    }
    fig = plt.figure()
    ax = fig.add_subplot()

    hist = trainer.state.log_history

    epoch_eval = [(x["epoch"], x.get(f"eval_{metric}", None)) for x in hist if
                  int(x["epoch"]) == x["epoch"]]
    epoch_eval = [x for x in epoch_eval if x[1] is not None]

    X = np.array([x for x, y in epoch_eval])

    Y = np.array([y * 100 for x, y in epoch_eval])

    ax.plot(X, Y, color=colors[metric])

    ax.plot(X[-1], Y[-1], "or")  # last point
    ax.axhline(y=Y[-1], color='r', linestyle='--')  # last eval line
    ax.annotate(round(Y[-1], 2), xy=(1, Y[-1] + 0.05), color='red')  # last eval annotate

    ind_max = np.argmax(Y)

    ax.plot(X[ind_max], Y[ind_max], "ob")  # last point
    ax.axhline(y=Y[ind_max], color='b', linestyle='--')  # last eval line
    ax.annotate(round(Y[ind_max], 2), xy=(1, Y[ind_max] + 0.05), color='blue')  # last eval annotate

    ax.set_title(title)
    ax.set_xlabel("epochs")
    ax.set_ylabel(f"eval {metric} %")

    fig.show()
    folder = f"Figures/{mode}/{file_name}"
    mkdir_p(folder)
    save_path = f"{folder}/{title}"
    fig.savefig(save_path)


def train_net(json_path, mode, limit, epoch_num, train_percent, val_percent, file_name):
    if os.path.exists('glove.npy'):
        print('glove.npy already exists')
    else:
        gdown.download('https://drive.google.com/uc?export=download&id=1PFOG06NEsTL6VieKQjMk1oNzyzcUtiWn', 'glove.npy',
                       quiet=False)
    if os.path.exists('vocab.json'):
        print('vocab.json already exists')
    else:
        gdown.download('https://drive.google.com/uc?export=download&id=1-3SxpirQjmX-RCRyRjKdP2L7G_tNgp00', 'vocab.json',
                       quiet=False)

    nltk.download('punkt')

    raw_datasets, nclasses = load_json(json_path, mode, limit, train_percent, val_percent)

    with open("vocab.json") as f:
        vocab = json.load(f)

    small_train_dataset = raw_datasets['train'].shuffle(seed=42).map(lambda x: tokenize_function(x, vocab),
                                                                     batched=True)
    small_eval_dataset = raw_datasets['test'].shuffle(seed=42).map(lambda x: tokenize_function(x, vocab), batched=True)

    co = DataCollatorWithPadding()
    training_args = TrainingArguments("DAN",
                                      num_train_epochs=epoch_num,
                                      per_device_train_batch_size=50,
                                      per_device_eval_batch_size=50,
                                      learning_rate=0.0005,

                                      save_total_limit=2,
                                      log_level="error",
                                      evaluation_strategy="epoch")
    model = DAN(nclasses)

    trainer = Trainer(
        model=model,
        data_collator=co,
        args=training_args,
        callbacks=[],
        train_dataset=small_train_dataset,
        eval_dataset=small_eval_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    for metric in ['accuracy', 'f1', 'precision', 'recall']:

        plot_trainer(trainer, f'epoch vs eval {metric}', metric, file_name, mode)


def main(dataset_name, tag_feature, limit):
    file_name = '{dataset_name}_Dataset_{limit}_{tag_feature}.json'.format(
        dataset_name=dataset_name, limit=limit, tag_feature=tag_feature)
    if tag_feature == 'original':
        path = './Datasets/{dataset_name}_Dataset_15000_upos.json'.format(
            dataset_name=dataset_name)
    else:
        path = './Datasets/{dataset_name}_Dataset_15000_{tag_feature}.json'.format(
            dataset_name=dataset_name, tag_feature=tag_feature)
        tag_feature = 'upos'
    train_net(path, tag_feature, limit, epoch_num=50, train_percent=0.8, val_percent=0, file_name=file_name)
