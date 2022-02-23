import torch
from torch import nn

class LSTM(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm1 = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        # self.lstm2 = nn.LSTM(hidden_dim, hidden_dim*2, batch_first=True)
        self.linear = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, l):
        x = self.embeddings(x)
        x = self.dropout(x)
        lstm_out1, (ht1, ct1) = self.lstm1(x)
        # lstm_out2, (ht2, ct2) = self.lstm2(lstm_out1)
        return self.linear(ht1[-1])