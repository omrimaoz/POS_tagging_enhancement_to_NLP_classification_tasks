from torch import nn
import torch
import numpy as np


def pad_sequence_to_length(
    sequence,
    desired_length: int,
    default_value=lambda: 0,
    padding_on_right: bool = True,
):
    sequence = list(sequence)
    # Truncates the sequence to the desired length.
    if padding_on_right:
        padded_sequence = sequence[:desired_length]
    else:
        padded_sequence = sequence[-desired_length:]
    # Continues to pad with default_value() until we reach the desired length.
    pad_length = desired_length - len(padded_sequence)
    # This just creates the default value once, so if it's a list, and if it gets mutated
    # later, it could cause subtle bugs. But the risk there is low, and this is much faster.
    values_to_pad = [default_value()] * pad_length
    if padding_on_right:
        padded_sequence = padded_sequence + values_to_pad
    else:
        padded_sequence = values_to_pad + padded_sequence
    return padded_sequence


class DAN(nn.Module):
    def __init__(self, class_num):
        super().__init__()
        self.num_labels = 2
        self.embeddings = nn.Embedding.from_pretrained(torch.FloatTensor(np.load("glove.npy")))

        self.input_size = 300
        self.hidden_size = 300
        self.output_size = class_num
        self.activation = nn.ReLU()
        self.input_layer = nn.Linear(self.input_size, self.hidden_size)
        self.hidden_layer = nn.Linear(self.hidden_size, self.hidden_size)
        self.output_layer = nn.Softmax()

        self.classifier = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.output_size)
        )

        self.loss = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_masks, labels=None, **kwargs):
        e = self.embeddings(input_ids)
        avg = torch.mean(e, 1)

        res = self.classifier(avg)
        loss = self.loss(res, labels)
        return {"loss": loss, "logits": res}
