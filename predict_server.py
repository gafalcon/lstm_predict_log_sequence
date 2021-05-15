import torch
from torch import nn
import numpy as np
from lstm import LSTM
from seq2seq import Seq2Seq
import matplotlib.pyplot as plt
import seaborn as sns
import config


class Prediction():

    def __init__(self, model_file, seq2seq_file):
        self.model_file = model_file
        self.seq2seq_file = seq2seq_file
        self.seq_types = np.array(config.seq_types)
        self.input_size = config.input_size
        self.output_size = config.output_size
        self.num_layers = config.num_layers
        self.batch_size = config.batch_size
        self.hidden_size = config.hidden_size
        self.seq = []
        self.seq_x = []
        self.init_seq = ['start']
        self.action = None

        self.load_models()

    def load_models(self):
        self.model = LSTM(self.input_size, self.output_size,
                          self.num_layers, batch_size=self.batch_size)
        self.seq2seq = Seq2Seq(self.input_size, self.output_size,
                             self.hidden_size, self.num_layers)
        self.model.load_state_dict(torch.load(f"{config.MODEL_PATH}/{self.model_file}"))
        self.seq2seq.load_state_dict(torch.load(f"{config.MODEL_PATH}/{self.seq2seq_file}"))
        self.model.eval()
        self.seq2seq.eval()
        return self.model

    def create_input(self, seq_type, method=None, url=None, delta=None, objectType=None):
        x = np.zeros(config.input_size)
        if method is None and url is None and delta is None:
            x[:4] = config.type_to_cat[config.seq_type_to_req_type[seq_type]]
            x[4+config.urls.index(config.seq_type_to_url[seq_type])] = 1
            x[-2] = 1.0
            x[-1] = 0.0
        else:
            x[:4] = config.type_to_cat[method]
            x[4+config.urls.index(url)] = 1
            x[-2] = delta
            x[-1] = config.objType_to_id.get(objectType, 0.0)
        return x

    def initialize_seq(self, init_seq=None):
        if init_seq is None:
            init_seq = self.init_seq
        self.seq = []
        self.seq_x = []
        self.action = None
        for a in init_seq:
            self.seq.append(a)
            new_input = self.create_input(a)
            self.seq_x.append(new_input)


    def predict_next_action(self, newAction=None, method=None,
                            url=None, delta=None, objType=None):
        if newAction is not None:
            self.action = newAction
        if self.action is not None:
            self.seq.append(self.action)
            new_input = self.create_input(self.action, method, url, delta, objType)
            self.seq_x.append(new_input)
        x = np.array(self.seq_x, dtype=float)
        batch_x = torch.Tensor(np.expand_dims(x, 0))

        #Predict
        out, state = self.model(batch_x)
        y_pred = out[0][-1]
        p = nn.functional.softmax(y_pred, dim=0).detach().numpy()
        sorted_idx = p.argsort()
        top_5_idx = sorted_idx[::-1][:5]
        top_5_p = p[top_5_idx]
        top_5_seq_types = self.seq_types[top_5_idx]
        predicted_seq_type =self.seq_types[p.argmax()]
        print("predicted: ", predicted_seq_type)
        prevAction = self.action
        self.action = predicted_seq_type
        return prevAction, top_5_seq_types, top_5_p

    def predict_sequence(self, sequence):
        sequence = torch.Tensor(np.expand_dims(sequence, 0))
        return self.seq2seq.predict(sequence)

# predict = Prediction("lstmv1", "seq2seq")
# seq2seq = predict.seq2seq
