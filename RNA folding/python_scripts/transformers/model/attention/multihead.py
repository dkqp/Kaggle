import torch

from .single import Attention


class MultiheadAttention(torch.nn.Module):
    def __init__(self, num_head, dim_k, dim_embed, dropout=0.1) -> None:
        super().__init__()

        self.num_head = num_head
        self.d_k = dim_k # dim_k equals dim_v
        self.dim_embed = dim_embed

        # 3 linear layers for q, k, v
        self.linear_layers = torch.nn.ModuleList([torch.nn.Linear(
            in_features = dim_embed,
            out_features = num_head * dim_k
        ) for _ in range(3)])

        self.output_linear_layer = torch.nn.Linear(
            in_features = num_head * dim_k,
            out_features = dim_embed
        )

        self.attention = Attention(dropout=dropout)

    def forward(self, q, k=None, v=None, mask=None, probs=None):
        '''
        q: query (batch * token_length * dim_embed)
        k: key (batch * token_length * dim_embed)
        v: value (batch * token_length * dim_embed)
        '''

        assert (k is None and v is None) or (k is not None)

        if v is None:
            if k is None:
                k = q
            v = k

        batch_size = q.size(0)

        q, k, v = [layer(x).view(batch_size, -1, self.num_head, self.d_k).transpose(1, 2) for layer, x in zip(self.linear_layers, [q, k, v])]
        # q, k, v: (batch, num_head, token_length, dim_k)

        x, attn = self.attention(q, k, v, mask=mask, probs=probs)
        # x: (batch, num_head, token_length, dim_v)

        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.num_head * self.d_k)
        x = self.output_linear_layer(x)

        return x
