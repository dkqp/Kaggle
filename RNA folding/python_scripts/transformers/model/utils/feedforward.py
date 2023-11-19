import torch


class FeedForward(torch.nn.Module):
    def __init__(self, dim_embed, dim_ff, dropout=0.1) -> None:
        super().__init__()

        self.layer1 = torch.nn.Linear(in_features=dim_embed, out_features=dim_ff)
        self.layer2 = torch.nn.Linear(in_features=dim_ff, out_features=dim_embed)
        self.dropout = torch.nn.Dropout(dropout)
        self.activation = torch.nn.GELU()

    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.dropout(x)
        return self.layer2(x)