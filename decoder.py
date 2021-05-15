import torch
from torch import nn

class SeqDecoder(nn.Module):

    def __init__(self, input_size, output_size, hidden_size, num_layers,dropout=0.5):
        super(SeqDecoder, self).__init__()
        self.output_size = output_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(self.input_size,
                            self.hidden_size,
                            num_layers=self.num_layers,
                            batch_first=True,
                            dropout=dropout
                           )
        self.out = nn.Linear(self.hidden_size, self.output_size)
        #self.softmax = nn.LogSoftmax(dim=1)
        #self.dropout = nn.Dropout(dropout)


    def forward(self, x, hidden, cell):
        x = x.unsqueeze(0)
        output, (hidden, cell) = self.lstm(x, (hidden, cell))
        prediction = self.out(output.squeeze(0))

        return prediction, hidden, cell
