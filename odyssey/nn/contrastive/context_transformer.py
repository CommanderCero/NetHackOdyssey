import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from typing import Optional

class TimestepEmbedding(nn.Module):
    def __init__(self,
        embedding_size: int,
        max_len: int = 100
    ):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_size, 2) * (-math.log(10000.0) / embedding_size))
        pe = torch.zeros(max_len, embedding_size)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.LongTensor) -> torch.Tensor:
        return self.pe[x]

class ContextTransformer(nn.Module):
    def __init__(self,
        input_size: int,
        output_size: int,
        num_blocks: int,
        hidden_size: int,
        max_trajectory_length: int,
        num_heads: int,
        drop_p: float=0.1
    ):
        super().__init__()
        self.input_size = input_size
        self.max_trajectory_length = max_trajectory_length

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_size,
            nhead=num_heads,
            dim_feedforward=hidden_size,
            dropout=drop_p,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_blocks
        )

        self.timestep_embedding = TimestepEmbedding(input_size, max_trajectory_length)
        self.embed_ln = nn.LayerNorm(input_size)
        self.output_fc = nn.Linear(input_size, output_size)

        # Initialize reusable causal mask
        # Not sure if we should include this for computing the context, since we aren't predicting anything, we just want to represent the past
        ones = torch.ones((max_trajectory_length, max_trajectory_length), dtype=bool)
        mask = torch.triu(ones, diagonal=1).view(max_trajectory_length, max_trajectory_length)
        self.register_buffer('mask', mask)
    
    def forward(self,
        observations: torch.Tensor,
        padding_mask: Optional[torch.Tensor]=None,
    ):
        B, T, E = observations.shape
        assert E == self.input_size, "Input size does not match the embedding size"
        assert T <= self.max_trajectory_length, "The number of observations exceeds the maximum trajectory length"

        # Add positional embeddings to the observations
        # TODO Cache torch.arange
        timestep_embeddings = self.timestep_embedding(torch.arange(T, device=observations.device)).unsqueeze(0).expand(B, T, E)
        input = observations + timestep_embeddings

        # Predict
        h = self.embed_ln(input)
        h = self.transformer(
            h,
            mask=self.mask[:T, :T],
            src_key_padding_mask=padding_mask,
            is_causal=True
        )

        # Compute embeddings for each trajectory
        if padding_mask is None:
            embeddings = torch.mean(h, dim=1)
        else:
            # True values in the padding mask indicate padding, so we need to invert it
            inv_pad = ~padding_mask
            embeddings = torch.sum(h * inv_pad.unsqueeze(-1), dim=1) / torch.sum(inv_pad, dim=1, keepdim=True)
        
        embeddings = self.output_fc(F.relu(embeddings))
        return embeddings