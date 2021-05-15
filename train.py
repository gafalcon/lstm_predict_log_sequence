import torch
import numpy as np
from torch import nn, optim
from lstm import LSTM
from lstm_v2 import LSTM2
from gru import GRU
from torch.utils.data import DataLoader
from kfDataset import SeqDataset
import config
import glob
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
        self.model_name = config.model
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

    def get_dataloader(self, train=True):
        if train:
            return DataLoader(self.train_dataset, batch_size=self.batch_size)
        else:
            return DataLoader(self.test_dataset, batch_size=self.batch_size)

    def eval(self, train=True):
        self.model.eval()
        dataloader = self.get_dataloader(train)
        with torch.no_grad():
            total_loss = 0.0
            for batch, (x, y, input_lens) in enumerate(dataloader):
                out, _ = self.model(x, input_lens=input_lens)
                y_pred = out[0]
                y_target = y[0, :input_lens[0]].long()
                loss = self.loss_fn(y_pred, y_target)
                # print(loss)
                total_loss += loss
            print("AVG loss: ", total_loss/(batch+1) )

    def load(self):
        self.dataset = SeqDataset(seq_len=self.seq_len)
        self.train_dataset = SeqDataset(seq_len=self.seq_len,
                                        datasets=glob.glob(config.train_data))
        self.test_dataset = SeqDataset(seq_len=self.seq_len,
                                       datasets=glob.glob(config.test_data))
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size)


    def train(self, epochs, dataloader=None):
        self.model.train()
        if dataloader is None:
            dataloader = self.dataloader
        for epoch in range(epochs):
            for batch, (x, y, input_lens) in enumerate(dataloader):
                state = self.model.init_state(self.batch_size)
                if x.shape[0] != self.batch_size:
                    continue
                self.optimizer.zero_grad()
                out, state = self.model(x, input_lens=input_lens, prev_state=state)
                y_pred = out[0]
                y_target = y[0, :input_lens[0]].long()
                loss = self.loss_fn(y_pred, y_target) #Take first seq in batch and convert y to long
                if self.model_name == "gru":
                    state = state.detach()
                else:
                    state = (state[0].detach(), state[1].detach())
                #state = (state_h.detach(), state_c.detach())
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

    def accuracy(self, train=True):
        self.model.eval()
        dataloader = self.get_dataloader(train)
        with torch.no_grad():
            accuracy = 0.0
            accuracy_top3 = 0.0
            total = 0
            for batch, (x, y, input_lens) in enumerate(dataloader):
                out, _ = self.model(x, input_lens=input_lens)
                y_pred = out[0]
                y_target = y[0, :input_lens[0]].long().detach().numpy()

                p = nn.functional.softmax(y_pred, dim=1).detach().numpy()
                top_3_idx = p.argsort(axis=1)[:, ::-1][:, :3]

                seq_len = int(input_lens[0])
                for i in range(int(seq_len/2), seq_len):
                    idx = y_target[i]
                    if idx in top_3_idx[i]:
                        accuracy_top3 += 1.0
                        if idx == top_3_idx[i, 0]:
                            accuracy += 1.0
                    total +=1
            accuracy = accuracy/ total
            accuracy_top3 = accuracy_top3/total
        return accuracy, accuracy_top3

    def save(self, filename):
        torch.save(self.model.state_dict(), filename)
        print("model saved in", filename)

config.parseArgs()
training = Training(config)
training.train(config.epochs, dataloader=training.get_dataloader(train=True))
training.eval(train=True)
print(training.accuracy(train=True))

training.eval(train=False)
print(training.accuracy(train=False))
if config.save:
    path = f"{config.MODEL_PATH}/{config.save}"
    training.save(path)
