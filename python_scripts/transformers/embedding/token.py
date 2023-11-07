import torch


class TokenEmbedding(torch.nn.Embedding):
    def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
        super().__init__(num_embeddings, embedding_dim, padding_idx=0)
