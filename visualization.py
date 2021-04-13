import torch
from lstm import LSTM
from lstm_v2 import LSTM2
from gru import GRU
import hiddenlayer as hl

input_size = 5
lstm = LSTM(input_size=input_size, num_layers=2, hidden_size=128)
lstm2 = LSTM2(input_size=input_size, num_layers=2, hidden_size=[128, 256])
gru = GRU(input_size=input_size, num_layers=2)

print(lstm)
print(lstm2)
print(gru)

batch_size = 1
seq_len = 15
input_dim = 5

transforms = [hl.transforms.Prune('Constant')]

# LSTM 1
x = torch.randn(batch_size, seq_len, input_dim)
hidden_state = lstm.init_state(batch_size)
y, _ = lstm(x, [seq_len]*batch_size, hidden_state)
print(y.shape)

graph = hl.build_graph(lstm, x, transforms=transforms)
graph.theme = hl.graph.THEMES['blue'].copy()
graph.save('lstm_hiddenlayer', format='png')

# LSTM 2
x = torch.randn(batch_size, seq_len, input_dim)
hidden_state = lstm2.init_state(batch_size)
y, _ = lstm2(x, [seq_len]*batch_size, hidden_state)
print(y.shape)

graph = hl.build_graph(lstm2, x, transforms=transforms)
graph.theme = hl.graph.THEMES['blue'].copy()
graph.save('lstm2_hiddenlayer', format='png')


# GRU
x = torch.randn(batch_size, seq_len, input_dim)
hidden_state = gru.init_state(batch_size)
y, _ = gru(x, [seq_len]*batch_size, hidden_state)
print(y.shape)

graph = hl.build_graph(gru, x, transforms=transforms)
graph.theme = hl.graph.THEMES['blue'].copy()
graph.save('gru_hiddenlayer', format='png')
