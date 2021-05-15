import torch
import numpy as np
from torch import nn, optim
from lstm import LSTM
from lstm_v2 import LSTM2
from gru import GRU
from torch.utils.data import DataLoader
from kfDataset import SeqDataset
import config

class Training():

    def __init__(self, config):
        self.seq_types = np.array(config.seq_types)
        self.seq_len = config.seq_len
        self.input_size = config.input_size
        self.output_size = config.output_size
        self.num_layers = config.num_layers
        self.batch_size = config.batch_size
        self.lr = config.lr
        self.epochs = config.epochs

        if config.model == "lstm1":
            self.model = LSTM(self.input_size, self.output_size,
                              self.num_layers, batch_size=self.batch_size)
        elif config.model == "lstm2":
            self.model = LSTM2(self.input_size, self.output_size, self.num_layers)
        elif config.model == "gru":
            self.model = GRU(self.input_size, self.output_size, self.num_layers)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss_fn = nn.CrossEntropyLoss()
        self.load()

    def eval(self):
        self.model.eval()
        with torch.no_grad():
            total_loss = 0.0
            for batch, (x, y, input_lens) in enumerate(self.dataloader):
                out, _ = self.model(x, input_lens=input_lens)
                y_pred = out[0]
                y_target = y[0, :input_lens[0]].long()
                loss = self.loss_fn(y_pred, y_target)
                # print(loss)
                total_loss += loss
            print("AVG loss: ", total_loss/(batch+1) )

    def load(self):
        self.dataset = SeqDataset(seq_len=self.seq_len)
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size)


    def train(self, epochs):
        self.model.train()
        for epoch in range(epochs):
            for batch, (x, y, input_lens) in enumerate(self.dataloader):
                state = self.model.init_state(self.batch_size)
                if x.shape[0] != self.batch_size:
                    continue
                self.optimizer.zero_grad()
                out, (state_h, state_c) = self.model(x, input_lens=input_lens, prev_state=state)
                y_pred = out[0]
                y_target = y[0, :input_lens[0]].long()
                loss = self.loss_fn(y_pred, y_target) #Take first seq in batch and convert y to long
                state = (state_h.detach(), state_c.detach())
                loss.backward()
                self.optimizer.step()

                print({'epoch': epoch, 'batch': batch, 'loss': loss.item()})
                if batch % 10 == 0:
                    p = nn.functional.softmax(y_pred, dim=1).detach().numpy()
                    print("sequence: ", self.seq_types[y_target])
                    print("predicted: ", self.seq_types[p.argmax(axis=1)])

            p = nn.functional.softmax(y_pred, dim=1).detach().numpy()
            print("sequence: ", self.seq_types[y_target])
            print("predicted: ", self.seq_types[p.argmax(axis=1)])

    def save(self, filename):
        torch.save(self.model.state_dict(), filename)
        print("model saved in", filename)

config.parseArgs()
training = Training(config)
training.train(config.epochs)
training.eval()
if config.save:
    path = f"{config.MODEL_PATH}/{config.save}"
    training.save(path)
