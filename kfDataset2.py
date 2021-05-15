import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import config
import glob

datasets = glob.glob("../datasets/kf/processed2/*csv")

class SeqDataset(Dataset):

    def __init__(self, seq_len=10, datasets=datasets):
        self.seq_len = seq_len
        self.x, self.y, self.out_lens  = [], [], []

        for filename in datasets:
            self.load_dataset(filename)

        # self.x = np.array(self.x, dtype=np.float32)
        # self.y = np.array(self.y, dtype=np.float32)

    def load_dataset(self, filename):
        df = pd.read_csv(filename)
        x = df.loc[:, config.x_columns].values.astype(np.float32)
        y = df.loc[df.seq_type.isin(config.predicted_seq_types), config.y_columns].values.astype(np.float32)
        # y = y.argmax(axis=1)
        self.x.append(x)
        self.y.append(y)
        self.out_lens.append(len(y))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.out_lens[idx]
