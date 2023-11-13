import torch
from torch.utils.data import Dataset

import pandas as pd
import numpy as np

class MaskedDataset(Dataset):
    def __init__(self, data: list[str], vocab: pd.DataFrame, max_len: int) -> None:
        super().__init__()

        self.seq_data = data['sequence']
        self.exp_type = data['experiment_type']
        self.vocab = vocab
        self.word_to_idx = {}
        for i in range(len(vocab)):
            self.word_to_idx[vocab['words'][i]] = i
        self.max_len = max_len

    def __len__(self):
        return len(self.seq_data)

    def __getitem__(self, index):
        data_added = [
            self.word_to_idx[self.exp_type[index]],
            self.word_to_idx['START'],
            *([self.word_to_idx[s] for s in self.seq_data[index]][:self.max_len - 3]),
            self.word_to_idx['END']
        ]
        data_added += [self.word_to_idx['PAD']] * (self.max_len - len(data_added))

        return torch.tensor(data_added)

class RNADataset(Dataset):
    def __init__(self, data: list[str], vocab: pd.DataFrame, max_len: int) -> None:
        super().__init__()

        self.seq_data = data['sequence']
        self.exp_type = data['experiment_type']

        label = []
        for i in range(1, 207):
            n = 4 - len(str(i))
            label.append(data[f"reactivity_{'0' * n + str(i)}"])
        self.label = np.array(label).transpose((1, 0))

        self.vocab = vocab
        self.word_to_idx = {}
        for i in range(len(vocab)):
            self.word_to_idx[vocab['words'][i]] = i
        self.max_len = max_len

    def __len__(self):
        return len(self.seq_data)

    def __getitem__(self, index):
        data_added = [
            self.word_to_idx[self.exp_type[index]],
            self.word_to_idx['START'],
            *([self.word_to_idx[s] for s in self.seq_data[index]][:self.max_len - 3]),
            self.word_to_idx['END']
        ]
        data_added += [self.word_to_idx['PAD']] * (self.max_len - len(data_added))

        label_added = np.array([
            -100,
            -100,
            *(self.label[index][:self.max_len - 2].clip(min=0, max=1)),
            -100
        ])
        label_added[np.isnan(label_added)] = -100
        label_added = np.concatenate([label_added, [-100] * (self.max_len - len(label_added))])

        return torch.tensor(data_added), torch.tensor(label_added)