import torch

from .encoder import EncoderBlock
from .embedding import CombEmbedding


class BERTCustom(torch.nn.Module):
    def __init__(self, vocab_size, hidden=768, dim_k=64, num_layer=12, num_attn_head=12, dropout=0.1) -> None:
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden = hidden
        self.num_layer = num_layer
        self.num_attn_head = num_attn_head
        self.dim_k = dim_k
        self.dim_ff = hidden * 4

        self.embedding = CombEmbedding(
            vocab_size=vocab_size,
            dim_embed=hidden,
            dropout=dropout
        )

        self.encoder_blocks = torch.nn.ModuleList([
            EncoderBlock(
                dim_embed=hidden,
                num_attn_head=num_attn_head,
                dim_k=dim_k,
                dim_ff=self.dim_ff,
                dropout=dropout
            ) for _ in range(num_layer)
        ])

    def forward(self, sequences):
        # sequences: (batch, len_tokens)
        mask = (sequences > 0).unsqueeze(1).repeat(1, sequences.size(1), 1).unsqueeze(1)
        # (batch, 1, len_tokens, len_tokens)

        x = self.embedding(sequences=sequences)

        for enc_block in self.encoder_blocks:
            x = enc_block(x, mask)

        return x