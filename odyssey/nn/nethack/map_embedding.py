from odyssey.nethack.constants import MAP_WIDTH, MAP_HEIGHT, BLSTATS_X_INDEX, BLSTATS_Y_INDEX
from odyssey.nn.crop import Crop2D

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchrl.modules as trl_modules

class MapEmbedding(nn.Module):
    def __init__(self,
        embedding_dim: int,
        input_glyph_embedding_dim: int,
        use_cropped_view=True,
        crop_size=9
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.glyph_embedding_dim = input_glyph_embedding_dim
        self.use_cropped_view = use_cropped_view

        self.conv_net = trl_modules.ConvNet(
            in_features=self.glyph_embedding_dim,
            depth=3,
            num_cells=16
        )

        self.embedding_proj = nn.LazyLinear(self.embedding_dim)

        if self.use_cropped_view:
            assert False, "Cropped view not implemented"
            self.crop = Crop2D(MAP_HEIGHT, MAP_WIDTH, crop_size, crop_size)
            self.cropped_view_embedding = nn.Sequential(
                nn.Conv2d(self.glyph_embedding_dim, 16, (3,3)),
                nn.ReLU(),
                nn.Conv2d(16, 16, (3,3)),
                nn.ReLU(),
                nn.Flatten(),
                nn.LazyLinear(self.embedding_dim),
            )

    def forward(self, glyphs: torch.LongTensor, blstats: torch.FloatTensor=None):
        # Input shape: [B x H x W x C]
        *B, _, _, _ = glyphs.shape
        glyphs = glyphs.permute(*range(len(B)), -1, -3, -2)

        embedding = self.conv_net(glyphs)
        embedding = self.embedding_proj(embedding)

        if self.use_cropped_view:
            assert blstats is not None, "blstats must be provided when using cropped view"
            x = blstats[..., BLSTATS_X_INDEX].long()
            y = blstats[..., BLSTATS_Y_INDEX].long()

            cropped_glyphs = self.crop(glyphs, x, y).long()
            embedding += self.cropped_view_embedding(cropped_glyphs)

        return embedding