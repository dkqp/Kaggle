import torch

from .bertcustom import BERTCustom


class BERTCustomRNAReactivity(torch.nn.Module):
    def __init__(self, bert: BERTCustom) -> None:
        super().__init__()

        self.bert = bert
        self.linear = torch.nn.Linear(in_features=bert.hidden, out_features=2)

    def forward(self, x):
        return self.linear(self.bert(sequences=x)).transpose(-1, -2)