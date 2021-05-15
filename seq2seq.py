import torch
from torch import nn
import numpy as np
import random
import config
from encoder import SeqEncoder
from decoder import SeqDecoder

class Seq2Seq(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers, dropout=0.5):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.encoder = SeqEncoder(input_size, hidden_size, num_layers, dropout=dropout)
        self.decoder = SeqDecoder(output_size, output_size, hidden_size, num_layers, dropout=dropout)

    def forward(self, x, y, seq_len, teacher_forcing_ratio = 0.5):

        # x = [ batch size, seq_len, input_size]
        # y = [ batch_size, seq_len, output_size]
        #teacher_forcing_ratio is probability to use teacher forcing
        batch_size = x.shape[0]
        #tensor to store decoder outputs
        outputs = torch.zeros(seq_len, batch_size, self.output_size)
        #last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, cell = self.encoder(x) # [num_layers, batch_size, hidden_size]
        #first input to the decoder is the <start> action
        input = y[:,0,:] # [1, output_size]

        for t in range(1, seq_len):
            #insert input action, previous hidden and previous cell states
            #receive output tensor (predictions) and new hidden and cell states
            predicted, hidden, cell = self.decoder(input, hidden, cell)
            #place predictions in a tensor holding predictions for each token
            outputs[t] = predicted #[1, output_size]
            #decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio
            #if teacher forcing, use actual next token as next input
            #if not, use predicted token
            input = y[:,t,:] if teacher_force else predicted

        return outputs

    def create_start_action(self):
        action = torch.zeros((1, self.output_size))
        action[config.predicted_seq_types.index('start')] = 1
        return action

    def predict(self, x, max_seq_len=30):
        self.eval()
        batch_size = x.shape[0]

        #outputs = torch.zeros(max_seq_len, batch_size, self.output_size)
        hidden, cell = self.encoder(x)
        predicted_seq_types = []
        probs = []
        seq_types = np.array(config.seq_types)
        input = self.create_start_action()
        for t in range(1, max_seq_len):
            predicted, hidden, cell = self.decoder(input, hidden, cell)
            #outputs[t] = predicted
            p = nn.functional.softmax(predicted[0], dim=0).detach().numpy()
            top_3_idx = p.argsort()[::-1][:3]
            top_3_p = list(p[top_3_idx].astype(float))
            top_3_actions = list(seq_types[top_3_idx])
            pred_action = seq_types[p.argmax().item()]
            predicted_seq_types.append(top_3_actions)
            probs.append(top_3_p)
            if pred_action == 'end':
                break
        return predicted_seq_types, probs
