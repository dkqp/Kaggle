import torch


class Attention(torch.nn.Module):
    def __init__(self, dropout=None) -> None:
        super().__init__()

        self.dropout = torch.nn.Dropout(p=dropout, inplace=True)

    def forward(self, q, k, v, mask=None):
        '''
        q: query (batch * num_head * token_length * q_dim)
        k: key (batch * num_head * token_length * k_dim)
        v: value (batch * num_head * token_length * v_dim)
        mask: tensor with 0 for masked position (token_length * token_length)
        ** assume q_dim == k_dim == v_dim == d_k
        '''
        attn_score = torch.matmul(q, torch.transpose(k, -1, -2)) / torch.sqrt(k.size(-1))
        # (batch, num_head, q_dim, k_dim)

        if mask is not None:
            attn_score = attn_score.masked_fill(mask == 0, value=1e-9)

        attn_p = torch.softmax(attn_score, dim=-1)
        attn_p = self.dropout(attn_p)

        attention = torch.matmul(attn_p, v)

        return attention, attn_p
