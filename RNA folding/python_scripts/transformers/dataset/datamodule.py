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
        self.num_workers = num_workers

        self.mask_token_ratio = mask_token_ratio
        self.mask_ratio = mask_ratio

        self.word_to_idx = whole_dataset.word_to_idx
        self.vocab_size = len(whole_dataset.vocab)

    def prepare_data(self) -> None:
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(self.train_val_dataset, self.train_val_test_ratio)

    def _collate_fn(self, batch):
        data = torch.stack(batch)
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

class RNADataModule(pl.LightningDataModule):
    def __init__(
            self,
            whole_train_dataset: Dataset = None,
            predict_dataset: Dataset = None,
            train_val_test_ratio: list[float] = [0.8, 0.1, 0.1],
            batch_size: int = 1,
            mask_token_ratio: float = 0.1,
            mask_ratio: list[float] = [0.8, 0.1, 0.1],
            augmentation: bool = False,
            sliced: bool = False,
            probs_adjusted: bool = False,
            num_workers: int = 0
        ):
        super().__init__()

        assert whole_train_dataset or predict_dataset
        assert sum(train_val_test_ratio) == 1
        assert 0 < mask_token_ratio < 1
        assert sum(mask_ratio) == 1

        assert (sliced == (batch_size == 1)) or not sliced
        assert sum([augmentation, sliced, probs_adjusted]) <= 1

        self.train_val_dataset = whole_train_dataset
        self.train_val_test_ratio = train_val_test_ratio
        self.predict_dataset = predict_dataset
        self.batch_size = batch_size

        self.augmentation = augmentation
        self.num_workers = num_workers

        self.mask_token_ratio = mask_token_ratio
        self.mask_ratio = mask_ratio

        self.sliced = sliced
        self.probs_adjusted = probs_adjusted

        if whole_train_dataset and hasattr(whole_train_dataset, 'word_to_idx'):
            self.word_to_idx = whole_train_dataset.word_to_idx

    def _collate_fn(self, batch):
        data = torch.stack([b[0] for b in batch])
        label = torch.stack([b[1] for b in batch])

        if self.probs_adjusted:
            probs = torch.stack([b[2] for b in batch])

            return data, label, probs

        if self.augmentation:
            aug_length = torch.randint(0, 300, ())
            aug_position = torch.randint(1, 207, ())

            data_added = torch.ones_like(data)[:, :aug_length] * self.word_to_idx['PAD']
            label_added = torch.ones_like(label)[:, :, :aug_length, :] * -100

            data = torch.concat([data[:, :aug_position], data_added, data[:, aug_position:]], dim=1)
            label = torch.concat([label[:, :, :aug_position, :], label_added, label[:, :, aug_position:, :]], dim=2)

            return data, label

        return data, label

    def _collate_fn_pred_sliced(self, batch):
        data = torch.stack([b[0] for b in batch])
        label = torch.stack([b[1] for b in batch])

        if self.sliced:
            return batch[0][0], batch[0][1]

        if self.probs_adjusted:
            probs = torch.stack([b[2] for b in batch])

            return data, label, probs

        return data, label

    def prepare_data(self) -> None:
        if self.train_val_dataset:
            self.train_dataset, self.val_dataset, self.test_dataset = random_split(self.train_val_dataset, self.train_val_test_ratio)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.train_dataset,
            self.batch_size,
            shuffle=True,
            collate_fn=self._collate_fn,
            num_workers=self.num_workers
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.val_dataset,
            self.batch_size,
            shuffle=False,
            collate_fn=self._collate_fn,
            num_workers=self.num_workers
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.test_dataset,
            self.batch_size,
            shuffle=False,
            collate_fn=self._collate_fn,
            num_workers=self.num_workers
        )

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            dataset=self.predict_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self._collate_fn_pred_sliced,
            num_workers=self.num_workers
        )