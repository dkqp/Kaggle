import torch

from .bertcustom import BERTCustom


class BERTCustomMasked(torch.nn.Module):
    def __init__(self, bert: BERTCustom) -> None:
        super().__init__()

        self.bert = bert
        self.linear = torch.nn.Linear(in_features=bert.hidden, out_features=bert.vocab_size)

    def forward(self, x):
        return self.linear(self.bert(sequences=x))