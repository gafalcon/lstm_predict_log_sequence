import torch
from torch import nn
import numpy as np
from lstm import LSTM
import matplotlib.pyplot as plt
import seaborn as sns
import config

model_file = "lstm.pth"
seq_types = np.array(config.seq_types)
input_size = config.input_size
output_size = config.output_size
num_layers = config.num_layers
batch_size = config.batch_size
model = LSTM(input_size, output_size, num_layers, batch_size=batch_size)
model.load_state_dict(torch.load(f"{config.MODEL_PATH}/{model_file}"))
model.eval()


def create_input(seq_type):
    x = np.zeros(config.input_size)
    x[:4] = config.type_to_cat[config.seq_type_to_req_type[seq_type]]
    x[4+config.urls.index(config.seq_type_to_url[seq_type])] = 1
    x[-1] = 1.0
    return x

def barplot(x, y, seq):
    ax = sns.barplot(x=x, y=y)
    ax.set(xlabel='action', ylabel='p', title=f"Seq: [{','.join(seq)}] : next action")
    plt.show()

seq = []
seq_x = []
action = 'start'
seq.append(action)
new_input = create_input(action)
seq_x.append(new_input)
x = np.array(seq_x, dtype=float)
batch_x = torch.Tensor(np.expand_dims(x, 0))

#Predict
out, state = model(batch_x)
y_pred = out[0]
p = nn.functional.softmax(y_pred, dim=1).detach().numpy()
sorted_idx = p.argsort(axis=1)
top_5_idx = sorted_idx[0][::-1][:5]
top_5_p = p[0][top_5_idx]
top_5_seq_types = seq_types[top_5_idx]
predicted_seq_type =seq_types[p.argmax(axis=1)]
barplot(top_5_seq_types, top_5_p, seq)
print("predicted: ", predicted_seq_type)
