import torch

from .bertcustom import BERTCustom


class BERTCustomRNAReactivity(torch.nn.Module):
    def __init__(self, bert: BERTCustom) -> None:
        super().__init__()

        self.bert = bert
        self.linear = torch.nn.Linear(in_features=bert.hidden, out_features=8)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.linear(self.bert(sequences=x))
        return x.view(batch_size, -1, 2, 4).transpose(1, 2)