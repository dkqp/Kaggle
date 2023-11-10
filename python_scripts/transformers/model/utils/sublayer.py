import torch


class SublayerResConnection(torch.nn.Module):
    def __init__(self, size, dropout) -> None:
        super().__init__()

        self.norm = torch.nn.LayerNorm(size)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return self.norm(x + self.dropout(sublayer(x)))