import torch

from .attention import MultiheadAttention
from .utils import FeedForward, SublayerResConnection


class EncoderBlock(torch.nn.Module):
    def __init__(self, dim_embed, num_attn_head, dim_k, dim_ff, dropout) -> None:
        super().__init__()

        self.attention = MultiheadAttention(
            num_head=num_attn_head,
            dim_k=dim_k,
            dim_embed=dim_embed,
            dropout=dropout
        )
        self.feedforward = FeedForward(dim_embed=dim_embed, dim_ff=dim_ff, dropout=dropout)
        self.sublayer1 = SublayerResConnection(size=dim_embed, dropout=dropout)
        self.sublayer2 = SublayerResConnection(size=dim_embed, dropout=dropout)
        self.dropout = torch.nn.Dropout(dropout, inplace=True)

    def forward(self, x, mask=None):
        x = self.sublayer1(x, lambda _x: self.attention(_x, _x, _x, mask=mask))
        x = self.sublayer2(x, self.feedforward)
        return self.dropout(x)