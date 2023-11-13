import torch

from .bertcustom import BERTCustom


class BERTCustomRNAReactivity(torch.nn.Module):
    def __init__(self, bert: BERTCustom) -> None:
        super().__init__()

        self.bert = bert
        self.linear = torch.nn.Linear(in_features=bert.hidden, out_features=1)
        self.activation = torch.nn.Sigmoid()

    def forward(self, x):
        return self.activation(self.linear(self.bert(sequences=x))).squeeze(-1)