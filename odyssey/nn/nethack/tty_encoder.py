from odyssey.nn.resnet import create_resnet
from odyssey.nn.nethack.tty_char_embedding import TTYCharEmbedding

import torch
import torch.nn as nn

from abc import ABC, abstractmethod

class TTYEncoderBase(nn.Module, ABC):
    def __init__(self, embedding_dim: int, char_embedding_dim: int = 16):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.char_embedding_dim = char_embedding_dim

    @abstractmethod
    def forward(self, tty_chars: torch.LongTensor, tty_colors: torch.LongTensor, tty_cursor: torch.LongTensor) -> torch.Tensor:
        """
        Input: (B, H, W) for tty_chars/colors, (B, 2) for cursor
        Output: (B, embedding_dim)
        """
        pass

    @abstractmethod
    def embed_tty_chars(self, tty_chars: torch.LongTensor, tty_colors: torch.LongTensor, tty_cursor: torch.LongTensor) -> torch.Tensor:
        """
        Input: (B, H, W), (B, H, W), (B, 2)
        Output: (B, H, W, char_embedding_dim)
        """
        pass

    @abstractmethod
    def encode_embeddings(self, char_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Input: (B, H, W, char_embedding_dim)
        Output: (B, embedding_dim)
        """
        pass

    def encode(
        self,
        tty_chars: torch.LongTensor = None,
        tty_colors: torch.LongTensor = None,
        tty_cursor: torch.LongTensor = None,
        char_embeddings: torch.Tensor = None
    ) -> torch.Tensor:
        if char_embeddings is not None:
            return self.encode_embeddings(char_embeddings)

        assert tty_chars is not None and tty_colors is not None and tty_cursor is not None, \
            "Must provide either char_embeddings or all of tty_chars, tty_colors, tty_cursor"

        *S, H, W = tty_chars.shape
        tty_chars = tty_chars.reshape(-1, H, W)
        tty_colors = tty_colors.reshape(-1, H, W)
        tty_cursor = tty_cursor.reshape(-1, 2)
        X = self.forward(tty_chars, tty_colors, tty_cursor)
        return X.reshape(*S, -1)

class ResnetTTYEncoder(TTYEncoderBase):
    def __init__(self,
        embedding_dim: int,
        char_embedding_dim: int = 16,
        resnet_type="resnet11"
    ):
        super().__init__(embedding_dim)
        self.embedding_dim = embedding_dim
        self.char_embedding_dim = char_embedding_dim

        self.chars_embedding = TTYCharEmbedding(char_embedding_dim)
        self.resnet = create_resnet(
            resnet_type,
            img_channels=char_embedding_dim,
            out_dim=embedding_dim
        )

    def forward(self, tty_chars: torch.LongTensor, tty_colors: torch.LongTensor, tty_cursor: torch.LongTensor) -> torch.Tensor:
        x = self.chars_embedding(tty_chars.long(), tty_colors.long(), tty_cursor.long())
        x = x.permute(0, 3, 1, 2).contiguous()  # For vmap compatibility, as otherwise I get errors
        x = self.resnet(x)
        return x
    
    def embed_tty_chars(self, tty_chars: torch.LongTensor, tty_colors: torch.LongTensor, tty_cursor: torch.LongTensor) -> torch.Tensor:
        *S, H, W = tty_chars.shape
        tty_chars = tty_chars.reshape(-1, H, W)
        tty_colors = tty_colors.reshape(-1, H, W)
        tty_cursor = tty_cursor.reshape(-1, 2)
        X = self.chars_embedding(tty_chars.long(), tty_colors.long(), tty_cursor.long())
        return X.reshape(*S, H, W, -1)

    def encode_embeddings(self, char_embeddings: torch.Tensor) -> torch.Tensor:
        *S, H, W, E = char_embeddings.shape
        char_embeddings = char_embeddings.reshape(-1, H, W, E)
        char_encodings = self.resnet(char_embeddings.permute(0, 3, 1, 2))
        char_encodings = char_encodings.reshape(*S, -1)
        return char_encodings