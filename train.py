import torch
import numpy as np
from torch import nn, optim
from lstm import LSTM
from torch.utils.data import DataLoader
from dataset import SeqDataset

seq_len = 10
input_size = 6
num_layers = 2
batch_size = 20
lr = 0.01
epochs = 5
dataset = SeqDataset(seq_len=seq_len)
model = LSTM(input_size, num_layers, batch_size=batch_size)
model.train()
dataloader = DataLoader(dataset, batch_size=batch_size)
optimizer = optim.Adam(model.parameters(), lr=lr)
loss_fn = nn.MSELoss()

for epoch in range(epochs):
    state = model.init_state(batch_size)
    for batch, (x, y) in enumerate(dataloader):
        if x.shape[0] != batch_size:
            continue
        optimizer.zero_grad()
        print(x.shape, y.shape, x.dtype)
        out, (state_h, state_c) = model(x, prev_state=state)
        y_pred = out[:, -1, :]
        loss = loss_fn(y_pred, y)
        state = (state_h.detach(), state_c.detach())
        loss.backward()
        optimizer.step()

        print({'epoch': epoch, 'batch': batch, 'loss': loss.item()})
