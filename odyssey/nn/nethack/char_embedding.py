from odyssey.nethack.constants import TTY_HEIGHT, TTY_WIDTH, NUM_TTY_CHARS, NUM_TTY_COLORS

import torch
import torch.nn as nn

class AdditiveCharEmbedding(nn.Module):
    """
    Embeds each character in the terminal of NetHack.
    This is a simple additive embedding that seperatly embeds tty_chars and tty_colors.
    The final embedding is the sum of these two embeddings, plus a tty_cursor embedding at the position of the cursor.
    """

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim

        self.tty_chars_embedding = nn.Embedding(NUM_TTY_CHARS, embedding_dim)
        self.tty_colors_embedding = nn.Embedding(NUM_TTY_COLORS, embedding_dim)
        self.cursor_embedding = nn.Parameter(torch.zeros(embedding_dim))
        nn.init.normal_(self.cursor_embedding)

    def forward(self, tty_chars: torch.LongTensor, tty_colors: torch.LongTensor, tty_cursor: torch.LongTensor) -> torch.Tensor:
        tty_chars_embedding = self.tty_chars_embedding(tty_chars)
        tty_colors_embedding = self.tty_colors_embedding(tty_colors)
        embedding = tty_chars_embedding + tty_colors_embedding

        # Add the cursor embedding at the position of the cursor
        y = tty_cursor[..., 0]
        x = tty_cursor[..., 1]
        embedding[torch.arange(len(embedding)), y, x] += self.cursor_embedding
        return embedding


