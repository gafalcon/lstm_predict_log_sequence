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
init_seq = ['start']
action = None

def initialize_seq(init_seq):
    global seq, seq_x
    seq = []
    seq_x = []
    for action in init_seq:
        seq.append(action)
        new_input = create_input(action)
        seq_x.append(new_input)


def predict_next_action(newAction=None):
    global action
    if newAction is not None:
        action = newAction
    if action is not None:
        seq.append(action)
        new_input = create_input(action)
        seq_x.append(new_input)
    x = np.array(seq_x, dtype=float)
    batch_x = torch.Tensor(np.expand_dims(x, 0))

    #Predict
    out, state = model(batch_x)
    y_pred = out[0][-1]
    p = nn.functional.softmax(y_pred, dim=0).detach().numpy()
    sorted_idx = p.argsort()
    top_5_idx = sorted_idx[::-1][:5]
    top_5_p = p[top_5_idx]
    top_5_seq_types = seq_types[top_5_idx]
    predicted_seq_type =seq_types[p.argmax()]
    barplot(top_5_seq_types, top_5_p, seq)
    print("sequence:", seq)
    print("predicted: ", predicted_seq_type)
    action = predicted_seq_type


initialize_seq(init_seq)
predict_next_action()
