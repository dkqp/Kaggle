import torch


class PositionEmbedding(torch.nn.Module):
    def __init__(self, dim_embed, max_len=1024) -> None:
        super().__init__()
        assert dim_embed % 2 == 0

        enc = torch.zeros(max_len, dim_embed, dtype=torch.float, requires_grad=False)

        position_num = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        position_denom = torch.exp(
            torch.log(torch.tensor(10000)) * torch.arange(0, dim_embed, 2, dtype=torch.float) / dim_embed
        )

        enc[:, 0::2] = torch.sin(position_num / position_denom)
        enc[:, 1::2] = torch.cos(position_num / position_denom)

        self.register_buffer('enc', enc.unsqueeze(0))
        # enc: (1, max_len, dim_embed)

    def forward(self, x):
        # x: sequences (batch, len_tokens)
        return self.enc[:, :x.size(1)]
