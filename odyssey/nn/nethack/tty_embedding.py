from odyssey.nn.resnet import create_resnet
from odyssey.nn.nethack.char_embedding import AdditiveCharEmbedding

import torch
import torch.nn as nn

from abc import ABC, abstractmethod

class TTYEmbeddingBase(nn.Module, ABC):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim

    # def encode(self, x):
    #     if isinstance(x, dict):
    #         return self.forward(x["tty_chars"], x["tty_colors"], x["tty_cursor"])
    #     elif isinstance(x, (list, tuple)) and len(x) == 3:
    #         return self.forward(*x)
    #     else:
    #         raise TypeError("Input must be a dict with keys 'tty_chars', 'tty_colors', 'tty_cursor', or a 3-tuple of tensors.")

    @abstractmethod
    def forward(self, tty_chars: torch.LongTensor, tty_colors: torch.LongTensor, tty_cursor: torch.LongTensor) -> torch.Tensor:
        raise NotImplementedError()

class ResnetTTYEmbedding(TTYEmbeddingBase):
    def __init__(self,
        embedding_dim: int,
        char_embedding_dim: int = 16,
        resnet_type="resnet11"
    ):
        super().__init__(embedding_dim)
        self.embedding_dim = embedding_dim
        self.char_embedding_dim = char_embedding_dim

        self.chars_embedding = AdditiveCharEmbedding(char_embedding_dim)
        self.resnet = create_resnet(
            resnet_type,
            img_channels=char_embedding_dim,
            out_dim=embedding_dim
        )

    def forward(self, tty_chars: torch.LongTensor, tty_colors: torch.LongTensor, tty_cursor: torch.LongTensor) -> torch.Tensor:
        x = self.chars_embedding(tty_chars.long(), tty_colors.long(), tty_cursor.long())
        x = x.permute(0, 3, 1, 2)
        x = self.resnet(x)
        return x