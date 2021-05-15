import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import config
import glob

datasets = glob.glob("../datasets/kf/processed2/*csv")

class SeqDataset(Dataset):

    def __init__(self, seq_len=10, datasets=datasets):
        self.seq_len = seq_len
        self.x, self.y, self.lens = [], [], []

        for filename in datasets:
            self.load_dataset(filename)

        self.x = np.array(self.x, dtype=np.float32)
        self.y = np.array(self.y, dtype=np.float32)

    def load_dataset(self, filename):
        df = pd.read_csv(filename)
        seq_len = min(len(df)-2, self.seq_len)
        x = df.loc[range(seq_len), config.x_columns].values
        y = df.loc[range(1, seq_len+1), config.y_columns].values
        y = y.argmax(axis=1)
        if seq_len < self.seq_len:
            x = np.concatenate(
                (x, np.zeros((self.seq_len - x.shape[0], config.input_size)))
            )
            y = np.concatenate(
                (y, np.zeros((self.seq_len - y.shape[0])))
            )
        self.x.append(x)
        self.y.append(y)
        self.lens.append(seq_len)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx, :self.seq_len], self.y[idx, :self.seq_len], self.lens[idx]
