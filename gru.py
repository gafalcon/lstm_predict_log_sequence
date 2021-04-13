import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class GRU(nn.Module):

    def __init__(self, input_size, output_size, num_layers=2, hidden_size=128):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.lstm = nn.GRU(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.drop = nn.Dropout(p=0.5)
        self.fc = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input_x, input_lens=None, prev_state=None):
        if prev_state is None:
            prev_state = self.init_state(input_x.shape[0])
        if input_lens is None:
            input_lens = [input_x.shape[1]]*input_x.shape[0]
        packed_input = pack_padded_sequence(
            input_x, input_lens, batch_first=True, enforce_sorted=False
        )
        packed_output, state = self.lstm(packed_input, prev_state)

        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        output = self.drop(output)  # Regularization

        output = self.fc(output)

        return output, state

    def init_state(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size)
