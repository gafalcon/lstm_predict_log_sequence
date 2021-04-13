import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LSTM2(nn.Module):

    def __init__(self, input_size, num_layers=2, hidden_size=[128, 256],
                 fc_size=128):
        super(LSTM2, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_layers = num_layers
        self.fc_size = fc_size
        self.lstm1 = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size[0],
            num_layers=self.num_layers,
            batch_first=True
        )
        self.lstm2 = nn.LSTM(
            input_size=self.hidden_size[0],
            hidden_size=self.hidden_size[1],
            num_layers=self.num_layers,
            batch_first=True
            )
        self.drop = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(self.hidden_size[-1], self.fc_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(self.fc_size, self.input_size)

    def forward(self, input_x, input_lens=None, prev_state=None):
        if prev_state is None:
            prev_state = self.init_state(input_x.shape[0])
        if input_lens is None:
            input_lens = [input_x.shape[1]]*input_x.shape[0]
        packed_input = pack_padded_sequence(
            input_x, input_lens, batch_first=True, enforce_sorted=False
        )
        packed_output, state1 = self.lstm1(packed_input, prev_state[0])

        packed_output, state2 = self.lstm2(packed_output, prev_state[1])

        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        output = self.drop(output)  # Regularization

        output = self.fc1(output)

        output = self.relu(output)

        output = self.fc2(output)

        return output, [state1, state2]

    def init_state(self, batch_size):
        return (
            (torch.zeros(self.num_layers, batch_size, self.hidden_size[0]),
             torch.zeros(self.num_layers, batch_size, self.hidden_size[0])),
            (torch.zeros(self.num_layers, batch_size, self.hidden_size[1]),
             torch.zeros(self.num_layers, batch_size, self.hidden_size[1]))
        )
