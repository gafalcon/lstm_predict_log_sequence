import torch
import numpy as np
from torch import nn, optim
from lstm import LSTM
from torch.utils.data import DataLoader
from kfDataset import SeqDataset
import config

seq_types = np.array(config.seq_types)
seq_len = 15
input_size = config.input_size
output_size = config.output_size
num_layers = config.num_layers
batch_size = config.batch_size
lr = 0.001
epochs = 50
dataset = SeqDataset(seq_len=seq_len)
model = LSTM(input_size, output_size, num_layers, batch_size=batch_size)
model.train()
dataloader = DataLoader(dataset, batch_size=batch_size)
optimizer = optim.Adam(model.parameters(), lr=lr)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(epochs):
    state = model.init_state(batch_size)
    for batch, (x, y, input_lens) in enumerate(dataloader):
        if x.shape[0] != batch_size:
            continue
        optimizer.zero_grad()
        out, (state_h, state_c) = model(x, input_lens=input_lens, prev_state=state)
        y_pred = out[0]
        y_target = y[0, :input_lens[0]].long()
        loss = loss_fn(y_pred, y_target) #Take first seq in batch and convert y to long
        state = (state_h.detach(), state_c.detach())
        loss.backward()
        optimizer.step()

        print({'epoch': epoch, 'batch': batch, 'loss': loss.item()})
    if epoch % 10 == 0:
        p = nn.functional.softmax(y_pred, dim=1).detach().numpy()
        # print(p)
        print("sequence: ", seq_types[y_target])
        print("predicted: ", seq_types[p.argmax(axis=1)])

p = nn.functional.softmax(y_pred, dim=1).detach().numpy()
# print(p)
print("sequence: ", seq_types[y_target])
print("predicted: ", seq_types[p.argmax(axis=1)])


#Save the trained model
torch.save(model.state_dict(), f"{config.MODEL_PATH}/lstm.pth")
