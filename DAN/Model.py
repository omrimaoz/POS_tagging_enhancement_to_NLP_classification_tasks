from torch import nn
import torch


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


# Use nn.Sequential and nn.Linear for the network, and nn.CrossEntropyLoss for the loss.
# Make sure that the final layer has output dimension of size 2.
class DAN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes):
        super().__init__()
        #   self.num_labels = num_classes
        self.embeddings = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=0)
        # YOUR CODE HERE
        self.input_size = embedding_dim
        self.hidden_size = hidden_dim
        self.output_size = num_classes
        self.activation = nn.ReLU()
        self.input_layer = nn.Linear(self.input_size, self.hidden_size)
        self.hidden_layer = nn.Linear(self.hidden_size, self.hidden_size)
        self.output_layer = nn.Softmax()

        self.classifier = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),

            # nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),

            # nn.Softmax()
            nn.Linear(self.hidden_size, self.output_size)
        )

        self.loss = nn.CrossEntropyLoss()
        # END YOUR END

    def forward(self, input_ids, labels=None, **kwargs):
        # YOUR CODE HERE
        e = self.embeddings(input_ids)
        avg = torch.mean(e, 1)
        # END YOUR END
        res = self.classifier(avg)
        print(res.shape, res)
        print(labels.shape, labels)
        loss = self.loss(res, labels)
        return {"loss": loss, "logits": res}
