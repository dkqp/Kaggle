import torch

from .token import TokenEmbedding
from .position import PositionEmbedding


class CombEmbedding(torch.nn.Module):
    def __init__(self, vocab_size, dim_embed, dropout=0.1) -> None:
        super().__init__()

        self.token = TokenEmbedding(num_embeddings=vocab_size, embedding_dim=dim_embed)
        self.position = PositionEmbedding(dim_embed=dim_embed)
        self.dropout = torch.nn.Dropout(dropout)
        self.dim_embed = dim_embed

    def forward(self, sequences):
        # sequences: (batch, len_tokens)
        x = self.token(sequences) + self.position(sequences)
        return self.dropout(x)