import torch
from torch import nn
import torch.nn.functional as F

class LSTM(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes):
        super().__init__()
        self.num_layers = 3
        self.num_classes = num_classes
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm1 = nn.LSTM(embedding_dim, hidden_dim, num_layers=self.num_layers, bidirectional=True, batch_first=True)
        # self.lstm2 = nn.LSTM(hidden_dim, hidden_dim//2, num_layers=self.num_layers, batch_first=True)
        # self.lstm3 = nn.LSTM(hidden_dim//2, hidden_dim//4, num_layers=self.num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.linear = nn.Linear(hidden_dim, 32)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x, l):
        x = self.embeddings(x)
        lstm_out1, (ht1, ct1) = self.lstm1(x)
        ht1 = self.relu(self.linear(ht1[-1]))
        ht1 = self.dropout(ht1)
        # lstm_out1 = self.dropout(lstm_out1)
        # lstm_out2, (ht2, ct2) = self.lstm2(lstm_out1)
        # lstm_out2 = self.dropout(lstm_out2)
        # lstm_out3, (ht3, ct3) = self.lstm3(lstm_out2)
        return self.fc(ht1)
