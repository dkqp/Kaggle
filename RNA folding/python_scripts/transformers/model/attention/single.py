import torch
import math


class Attention(torch.nn.Module):
    def __init__(self, dropout=None) -> None:
        super().__init__()

        self.dropout = torch.nn.Dropout(p=dropout)
        self.alpha = torch.nn.Parameter(torch.tensor(1, dtype=torch.float))

    def forward(self, q, k, v, mask=None, probs=None):
        '''
        q: query (batch * num_head * token_length * q_dim)
        k: key (batch * num_head * token_length * k_dim)
        v: value (batch * num_head * token_length * v_dim)
        mask: tensor with 0 for masked position (batch * token_length * token_length)
        ** assume q_dim == k_dim == v_dim == d_k

        probs: tensor of probabilities between each token (token_length * token_length)
        '''

        attn_score = torch.matmul(q, torch.transpose(k, -1, -2)) / math.sqrt(k.size(-1))
        if probs is not None:
            attn_score += probs.unsqueeze(1) * self.alpha
        # (batch, num_head, token_length, token_length)

        if mask is not None:
            attn_score = attn_score.masked_fill(mask == 0, value=-float('inf'))

        attn_p = torch.softmax(attn_score, dim=-1)
        attn_p = self.dropout(attn_p)
        # (batch, num_head, token_length, 1)

        attention = torch.matmul(attn_p, v)
        # (batch, num_head, token_length, v_dim)

        return attention, attn_p
