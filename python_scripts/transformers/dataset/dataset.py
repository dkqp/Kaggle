from torch.utils.data import Dataset

import pandas as pd
import numpy as np

class MaskedDataset(Dataset):
    def __init__(self, data: list[str], vocab: pd.DataFrame, max_len: int) -> None:
        super().__init__()

        self.data = data
        self.vocab = vocab
        self.word_to_idx = {}
        for i in range(len(vocab)):
            self.word_to_idx[vocab['words'][i]] = i
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_added = [
            self.word_to_idx['START'],
            *([self.word_to_idx[s] for s in self.data[index]][:self.max_len - 2]),
            self.word_to_idx['END']
        ]
        data_added += [self.word_to_idx['PAD']] * (self.max_len - len(data_added))

        return data_added

class RNADataset(Dataset):
    def __init__(self, data: list[str], label: list, vocab: pd.DataFrame, max_len: int) -> None:
        super().__init__()

        self.data = data
        self.label = label
        self.vocab = vocab
        self.word_to_idx = {}
        for i in range(len(vocab)):
            self.word_to_idx[vocab['words'][i]] = i
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_added = [
            self.word_to_idx['START'],
            *([self.word_to_idx[s] for s in self.data[index]][:self.max_len - 2]),
            self.word_to_idx['END']
        ]
        data_added += [self.word_to_idx['PAD']] * (self.max_len - len(data_added))

        label_added = np.array([
            0,
            *(self.label[index][:self.max_len - 2]),
            0
        ])
        label_added[np.isnan(label_added)] = 0
        label_added = np.concatenate([label_added, [0] * (self.max_len - len(label_added))])

        return data_added, label_added