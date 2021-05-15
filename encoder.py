import torch
from torch import nn

class SeqEncoder(nn.Module):

    def __init__(self, input_size, hidden_size,
                 num_layers, dropout=0.5):
        super(SeqEncoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(self.input_size,
                          self.hidden_size,
                          num_layers=self.num_layers,
                          batch_first=True,
                           dropout=dropout)
        #self.dropout = nn.Dropout(dropout)

    def forward(self, input_x):
       outputs, (hidden, cell) = self.lstm(input_x)
       return hidden, cell
