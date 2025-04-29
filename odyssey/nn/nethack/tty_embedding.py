from odyssey.nn.resnet import create_resnet
from odyssey.nn.nethack.char_embedding import AdditiveCharEmbedding

import torch
import torch.nn as nn

class ResnetTTYEmbedding(nn.Module):
    def __init__(self,
        embedding_dim: int,
        char_embedding_dim: int = 16,
        resnet_type="resnet11"
    ):
        super().__init__()
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