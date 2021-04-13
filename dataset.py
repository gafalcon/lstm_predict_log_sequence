import pickle
import numpy as np
from torch.utils.data import Dataset


class SeqDataset(Dataset):

    def __init__(self, seq_len=10):
        self.seq_len = seq_len
        sessions = pickle.load(open('./datasets/CTI-Data/cti_sessions.p', "rb"))
        self.x, self.y = [], []
        for session in sessions:
            self.split_sequence(session, self.seq_len, self.x, self.y)
        self.x = np.array(self.x, dtype=np.float32)
        self.y = np.array(self.y, dtype=np.float32)

    def split_sequence(self, sequence, seq_len, x, y):
        for i in range(len(sequence)):
            end_ix = i + seq_len
            if end_ix > len(sequence)-1:
                break

            seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
            x.append(np.squeeze(seq_x))
            y.append(np.squeeze(seq_y))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
