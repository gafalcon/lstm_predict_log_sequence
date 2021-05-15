import torch
from torch import nn, optim
import numpy as np
from torch.utils.data import DataLoader
import glob
import config
from kfDataset2 import SeqDataset
from seq2seq import Seq2Seq


class TrainSeq2Seq:

    def __init__(self, config):
        self.seq_types = np.array(config.seq_types)
        self.seq_len = config.seq_len
        self.input_size = config.input_size
        self.output_size = config.output_size
        self.num_layers = config.num_layers
        self.batch_size = config.batch_size
        self.hidden_size = config.hidden_size
        self.lr = config.lr
        self.epochs = config.epochs

        self.model = Seq2Seq(self.input_size, self.output_size,
                             self.hidden_size, self.num_layers)


        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss_fn = nn.CrossEntropyLoss()
        self.load()


    def load(self):
        self.dataset = SeqDataset(seq_len=self.seq_len)
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size)
        self.train_dataset = SeqDataset(seq_len=self.seq_len,
                                        datasets=glob.glob(config.train_data))
        self.test_dataset = SeqDataset(seq_len=self.seq_len,
                                       datasets=glob.glob(config.test_data))


    def train(self, epochs, dataloader=None):
        self.model.train()
        if dataloader is None:
            dataloader = self.dataloader

        # x1, y, input_lens = iter(self.dataloader).next()
        for epoch in range(epochs):
            for batch, (x, y, output_lens) in enumerate(dataloader):
                # x = x[:,:output_lens.item(),:]
                self.optimizer.zero_grad()
                output = self.model(x, y, output_lens.item())
                y_pred = output[1:, 0, :]
                y_target = y[0, 1:output_lens.item()].argmax(axis=1)
                loss = self.loss_fn(y_pred, y_target) #Take first seq in batch and convert y to long
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
            for batch, (x, y, out_lens) in enumerate(dataloader):
                output = self.model(x, y, out_lens.item())
                y_pred = output[1:, 0, :]
                y_target = y[0, 1:out_lens.item()].argmax(axis=1)
                loss = self.loss_fn(y_pred, y_target)
                total_loss += loss
            print("AVG loss: ", total_loss/(batch+1) )

    def accuracy(self, train=True):
        dataloader = self.get_dataloader(train)
        self.model.eval()
        with torch.no_grad():
            accuracy = 0.0
            accuracy_top3 = 0.0
            total = 0
            for batch, (x, y, out_lens) in enumerate(dataloader):
                y_target = y[0, 1:out_lens.item()].argmax(axis=1)
                y_target_seq = self.seq_types[y_target]
                predicted_seq_types, probs = self.model.predict(x)
                for i,target_action in enumerate(y_target_seq):
                    if len(predicted_seq_types) > i and target_action in predicted_seq_types[i]:
                        accuracy_top3 += 1.0
                        if target_action == predicted_seq_types[i][0]:
                            accuracy +=1.0
                    total += 1
            accuracy = accuracy/ total
            accuracy_top3 = accuracy_top3/total
        return accuracy, accuracy_top3



config.parseArgs()
training = TrainSeq2Seq(config)
training.train(config.epochs, dataloader=training.get_dataloader(train=True))
training.eval(train=True)
training.eval(train=False)
print(training.accuracy(train=True))
print(training.accuracy(train=False))
if config.save:
    path = f"{config.MODEL_PATH}/{config.save}"
    training.save(path)
