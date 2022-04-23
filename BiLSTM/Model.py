import torch
from torch import nn

class BiLSTM(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes):
        super().__init__()
        self.num_layers = 3
        self.num_classes = num_classes
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=self.num_layers, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.linear = nn.Linear(hidden_dim, 32)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x, l):
        x = self.embeddings(x)
        lstm_out, (ht, ct) = self.lstm(x)
        ht = self.relu(self.linear(ht[-1]))
        ht = self.dropout(ht)
        return self.fc(ht)
