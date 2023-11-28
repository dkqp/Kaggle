import torch
from torch.utils.data import Dataset

import pandas as pd
import numpy as np

class MaskedDataset(Dataset):
    def __init__(self, data_ext: pd.DataFrame, vocab: pd.DataFrame, max_len: int) -> None:
        super().__init__()

        self.data_ext = data_ext

        self.vocab = vocab
        self.word_to_idx = {}
        for i in range(len(vocab)):
            self.word_to_idx[vocab['words'][i]] = i
        self.max_len = max_len

    def __len__(self):
        return len(self.data_ext)

    def __getitem__(self, index):
        data_added = [
            self.word_to_idx['START'],
            *([self.word_to_idx[s + e] for s, e in zip(self.data_ext.iloc[index]['sequence'], self.data_ext.iloc[index]['sequence_ext'])][:self.max_len - 2]),
            self.word_to_idx['END']
        ]
        data_added += [self.word_to_idx['PAD']] * (self.max_len - len(data_added))

        return torch.tensor(data_added)

class RNADataset_train(Dataset):
    def __init__(self, data: pd.DataFrame, data_ext: pd.DataFrame, vocab: pd.DataFrame, max_len: int) -> None:
        super().__init__()

        self.data_ext = data_ext

        label = []
        for i in range(1, 207):
            n = 4 - len(str(i))
            label.append(data[f"reactivity_{'0' * n + str(i)}"])
        label = np.array(label).transpose((1, 0))
        self.label = np.array([label[0::2], label[1::2]])

        self.vocab = vocab
        self.word_to_idx = {}
        for i in range(len(vocab)):
            self.word_to_idx[vocab['words'][i]] = i
        self.max_len = max_len

    def __len__(self):
        return len(self.data_ext)

    def __getitem__(self, index):
        data_added = [
            self.word_to_idx['START'],
            *([self.word_to_idx[s + e] for s, e in zip(self.data_ext.iloc[index]['sequence'], self.data_ext.iloc[index]['sequence_ext'])][:self.max_len - 2]),
            self.word_to_idx['END']
        ]
        data_added += [self.word_to_idx['PAD']] * (self.max_len - len(data_added))
        # size: (self.max_len)

        data_added = np.array(data_added)


        label_mask = np.array([
            np.where((data_added != self.word_to_idx['A(']) & (data_added != self.word_to_idx['A)']) & (data_added != self.word_to_idx['A.']), True, False),
            np.where((data_added != self.word_to_idx['G(']) & (data_added != self.word_to_idx['G)']) & (data_added != self.word_to_idx['G.']), True, False),
            np.where((data_added != self.word_to_idx['C(']) & (data_added != self.word_to_idx['C)']) & (data_added != self.word_to_idx['C.']), True, False),
            np.where((data_added != self.word_to_idx['U(']) & (data_added != self.word_to_idx['U)']) & (data_added != self.word_to_idx['U.']), True, False),
        ]).transpose((1, 0))
        # size: (self.max_len, 4)

        label_added = np.expand_dims(self.label[:, index], axis=-1).clip(min=0, max=1)
        # size: (2, 206, 1)

        label_added = label_added.repeat(4, axis=-1)
        # size: (2, 206, 4)

        label_added[np.isnan(label_added)] = -100

        label_added = np.concatenate([
            np.ones((2, 1, 4)) * -100,
            label_added,
            np.ones((2, self.max_len - label_added.shape[1] - 1, 4)) * -100
        ], axis=1)
        # size: (2, self.max_len, 4)

        label_added[:, label_mask] = -100


        return torch.tensor(data_added), torch.tensor(label_added)

class RNADataset_pred(Dataset):
    def __init__(self, data_ext: pd.DataFrame, vocab: pd.DataFrame, max_len: int) -> None:
        super().__init__()

        self.data_ext = data_ext

        self.vocab = vocab
        self.word_to_idx = {}
        for i in range(len(vocab)):
            self.word_to_idx[vocab['words'][i]] = i
        self.max_len = max_len

    def __len__(self):
        return len(self.data_ext)

    def __getitem__(self, index):
        data_added = [
            self.word_to_idx['START'],
            *([self.word_to_idx[s + e] for s, e in zip(self.data_ext.iloc[index]['sequence'], self.data_ext.iloc[index]['sequence_ext'])][:self.max_len - 2]),
            self.word_to_idx['END']
        ]
        data_added += [self.word_to_idx['PAD']] * (self.max_len - len(data_added))

        data_added = np.array(data_added)

        label_mask = np.expand_dims(np.array([
            np.where((data_added == self.word_to_idx['A(']) | (data_added == self.word_to_idx['A)']) | (data_added == self.word_to_idx['A.']), True, False),
            np.where((data_added == self.word_to_idx['G(']) | (data_added == self.word_to_idx['G)']) | (data_added == self.word_to_idx['G.']), True, False),
            np.where((data_added == self.word_to_idx['C(']) | (data_added == self.word_to_idx['C)']) | (data_added == self.word_to_idx['C.']), True, False),
            np.where((data_added == self.word_to_idx['U(']) | (data_added == self.word_to_idx['U)']) | (data_added == self.word_to_idx['U.']), True, False),
        ]).transpose((1, 0)), axis=0).repeat(2, axis=0)
        # size: (2, self.max_len, 4)

        return torch.tensor(data_added), torch.tensor(label_mask)

class RNAdataset_sliced_train(Dataset):
    def __init__(self, data: torch.Tensor, vocab: pd.DataFrame) -> None:
        super().__init__()

        self.data = data
        self.vocab = vocab

    def __len__(self):
        return len(self.data['inputs'])

    def __getitem__(self, index):
        return self.data['inputs'][index], self.data['labels'][index]

class RNADataset_sliced_pred(Dataset):
    def __init__(self, data_ext: pd.DataFrame, vocab: pd.DataFrame, max_len: int) -> None:
        super().__init__()

        self.data_ext = data_ext

        self.vocab = vocab
        self.word_to_idx = {}
        for i in range(len(vocab)):
            self.word_to_idx[vocab['words'][i]] = i
        self.max_len = max_len

    def __len__(self):
        return len(self.data_ext)

    def __getitem__(self, index):
        data_added = make_sliced_pred(
            data=self.data_ext.iloc[index]['sequence'],
            data_ext=self.data_ext.iloc[index]['sequence_ext'],
            vocab=self.vocab
        )

        data_added = np.array(data_added)

        label_mask = np.expand_dims(np.array([
            np.where((data_added == self.word_to_idx['A(']) | (data_added == self.word_to_idx['A)']) | (data_added == self.word_to_idx['A.']), True, False),
            np.where((data_added == self.word_to_idx['G(']) | (data_added == self.word_to_idx['G)']) | (data_added == self.word_to_idx['G.']), True, False),
            np.where((data_added == self.word_to_idx['C(']) | (data_added == self.word_to_idx['C)']) | (data_added == self.word_to_idx['C.']), True, False),
            np.where((data_added == self.word_to_idx['U(']) | (data_added == self.word_to_idx['U)']) | (data_added == self.word_to_idx['U.']), True, False),
        ]).transpose((1, 2, 0)), axis=1).repeat(2, axis=1)
        # size: (len(data_added), 2, self.max_len, 4)

        return torch.tensor(data_added), torch.tensor(label_mask)

def make_sliced_pred(data: str, data_ext: str, vocab: pd.DataFrame):
    word_to_idx = {}
    for i in range(len(vocab)):
        word_to_idx[vocab['words'][i]] = i

    token_sliced = []
    j = 0
    while j + 50 < len(data):
        token_sliced.append([word_to_idx[s + e] for s, e in zip(data[j:j+100], data_ext[j:j+100])])
        j += 50

    if len(token_sliced[-1]) < 100:
        token_sliced[-1] += [word_to_idx['PAD']] * (100 - len(token_sliced[-1]))

    return token_sliced

class RNADataset_probs_train(Dataset):
    def __init__(self, data: pd.DataFrame, data_ext: pd.DataFrame, path_probs: dict[str], vocab: pd.DataFrame, max_len: int) -> None:
        super().__init__()

        self.data = data
        self.data_ext = data_ext
        self.path_probs = path_probs

        label = []
        for i in range(1, 207):
            n = 4 - len(str(i))
            label.append(data[f"reactivity_{'0' * n + str(i)}"])
        label = np.array(label).transpose((1, 0))
        self.label = np.array([label[0::2], label[1::2]])

        self.vocab = vocab
        self.word_to_idx = {}
        for i in range(len(vocab)):
            self.word_to_idx[vocab['words'][i]] = i
        self.max_len = max_len

    def __len__(self):
        return len(self.data_ext)

    def __getitem__(self, index):
        data_added = [
            self.word_to_idx['START'],
            *([self.word_to_idx[s + e] for s, e in zip(self.data_ext.iloc[index]['sequence'], self.data_ext.iloc[index]['sequence_ext'])][:self.max_len - 2]),
            self.word_to_idx['END']
        ]
        data_added += [self.word_to_idx['PAD']] * (self.max_len - len(data_added))
        # size: (self.max_len)

        data_added = np.array(data_added)


        label_mask = np.array([
            np.where((data_added != self.word_to_idx['A(']) & (data_added != self.word_to_idx['A)']) & (data_added != self.word_to_idx['A.']), True, False),
            np.where((data_added != self.word_to_idx['G(']) & (data_added != self.word_to_idx['G)']) & (data_added != self.word_to_idx['G.']), True, False),
            np.where((data_added != self.word_to_idx['C(']) & (data_added != self.word_to_idx['C)']) & (data_added != self.word_to_idx['C.']), True, False),
            np.where((data_added != self.word_to_idx['U(']) & (data_added != self.word_to_idx['U)']) & (data_added != self.word_to_idx['U.']), True, False),
        ]).transpose((1, 0))
        # size: (self.max_len, 4)

        label_added = np.expand_dims(self.label[:, index], axis=-1).clip(min=0, max=1)
        # size: (2, 206, 1)

        label_added = label_added.repeat(4, axis=-1)
        # size: (2, 206, 4)

        label_added[np.isnan(label_added)] = -100

        label_added = np.concatenate([
            np.ones((2, 1, 4)) * -100,
            label_added,
            np.ones((2, self.max_len - label_added.shape[1] - 1, 4)) * -100
        ], axis=1)
        # size: (2, self.max_len, 4)

        label_added[:, label_mask] = -100


        probs = make_probs_map(
            target_path=self.path_probs[self.data.iloc[index * 2]['sequence_id']],
            max_length=self.max_len
        )


        return torch.tensor(data_added), torch.tensor(label_added), probs

class RNADataset_probs_pred(Dataset):
    def __init__(self, data: pd.DataFrame, data_ext: pd.DataFrame, path_probs: dict[str], vocab: pd.DataFrame, max_len: int) -> None:
        super().__init__()

        self.data = data
        self.data_ext = data_ext
        self.path_probs = path_probs

        self.vocab = vocab
        self.word_to_idx = {}
        for i in range(len(vocab)):
            self.word_to_idx[vocab['words'][i]] = i
        self.max_len = max_len

    def __len__(self):
        return len(self.data_ext)

    def __getitem__(self, index):
        data_added = [
            self.word_to_idx['START'],
            *([self.word_to_idx[s + e] for s, e in zip(self.data_ext.iloc[index]['sequence'], self.data_ext.iloc[index]['sequence_ext'])][:self.max_len - 2]),
            self.word_to_idx['END']
        ]
        data_added += [self.word_to_idx['PAD']] * (self.max_len - len(data_added))

        data_added = np.array(data_added)


        label_mask = np.expand_dims(np.array([
            np.where((data_added == self.word_to_idx['A(']) | (data_added == self.word_to_idx['A)']) | (data_added == self.word_to_idx['A.']), True, False),
            np.where((data_added == self.word_to_idx['G(']) | (data_added == self.word_to_idx['G)']) | (data_added == self.word_to_idx['G.']), True, False),
            np.where((data_added == self.word_to_idx['C(']) | (data_added == self.word_to_idx['C)']) | (data_added == self.word_to_idx['C.']), True, False),
            np.where((data_added == self.word_to_idx['U(']) | (data_added == self.word_to_idx['U)']) | (data_added == self.word_to_idx['U.']), True, False),
        ]).transpose((1, 0)), axis=0).repeat(2, axis=0)
        # size: (2, self.max_len, 4)


        probs = make_probs_map(
            target_path=self.path_probs[self.data.iloc[index * 2]['sequence_id']],
            max_length=self.max_len
        )

        return torch.tensor(data_added), torch.tensor(label_mask), probs

def make_probs_map(target_path: str, max_length: int):
    '''
    Fill probs of [max_length, max_length] from index (1) to index (len(sequence))
    '''
    with open(target_path, 'r') as file:
        prob_matrix = torch.zeros((len(max_length), len(max_length)))
        for line in file:
            values = line.strip().split()
            if len(values) < 3:
                continue
            index1 = int(values[0])
            index2 = int(values[1])
            prob = float(values[2])
            prob_matrix[index1, index2] = prob
            prob_matrix[index2, index1] = prob

    return prob_matrix