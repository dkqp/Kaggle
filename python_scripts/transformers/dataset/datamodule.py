import torch
from torch.utils.data import DataLoader, Dataset, random_split

import lightning.pytorch as pl
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS

import numpy as np


class MaskedDataModule(pl.LightningDataModule):
    def __init__(
            self,
            whole_dataset: Dataset,
            train_val_test_ratio: list[float] = [0.8, 0.1, 0.1],
            batch_size: int = 1,
            shuffle: bool = False,
            mask_token_ratio: float = 0.1,
            mask_ratio: list[float] = [0.8, 0.1, 0.1],
            num_workers: int = 0
        ):
        super().__init__()

        assert sum(train_val_test_ratio) == 1
        assert 0 < mask_token_ratio < 1
        assert sum(mask_ratio) == 1

        self.train_val_dataset = whole_dataset
        self.train_val_test_ratio = train_val_test_ratio
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers

        self.mask_token_ratio = mask_token_ratio
        self.mask_ratio = mask_ratio

        self.word_to_idx = whole_dataset.word_to_idx
        self.vocab_size = len(whole_dataset.vocab)

    def prepare_data(self) -> None:
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(self.train_val_dataset, self.train_val_test_ratio)

    def _collate_fn(self, batch):
        data = torch.tensor(batch)
        label = torch.empty_like(data)

        for i in range(len(data)):
            token_len = 0
            for token in data[i]:
                if token != self.word_to_idx['PAD']:
                    token_len += 1
                    continue
                break

            masked_tokens = torch.tensor(np.random.choice(range(
                1, token_len - 1), int((token_len - 2) * self.mask_token_ratio)
            ))
            random_tokens = torch.randint_like(masked_tokens, 1, 101)

            label[i][:] = -100
            label[i][masked_tokens] = data[i][masked_tokens]
            data[i][masked_tokens] = torch.where(
                random_tokens <= int(self.mask_ratio[0] * 100),
                self.word_to_idx['MASK'],
                torch.where(
                    random_tokens <= int((1 - self.mask_ratio[2]) * 100),
                    torch.randint(0, self.vocab_size, (len(masked_tokens), )),
                    data[i][masked_tokens]
                )
            )

        return data, label

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.train_dataset,
            self.batch_size,
            collate_fn=self._collate_fn,
            shuffle=True,
            num_workers=self.num_workers
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.val_dataset,
            self.batch_size,
            collate_fn=self._collate_fn,
            shuffle=False,
            num_workers=self.num_workers
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.test_dataset,
            self.batch_size,
            collate_fn=self._collate_fn,
            shuffle=False,
            num_workers=self.num_workers
        )

