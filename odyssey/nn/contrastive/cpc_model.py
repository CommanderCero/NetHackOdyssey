from odyssey.nn.utils import apply_model_to_sequence

import torch
import torch.nn as nn

class CPCModel(nn.Module):
    def __init__(self,
            context_length: int,
            future_length: int,
            obs_embedding_dim: int,
            context_embedding_dim: int,
            obs_embedding: nn.Module,
            context_embedding: nn.Module,
            future_obs_predictor: nn.Module,
        ):
        super(CPCModel, self).__init__()

        self.context_length = context_length
        self.future_length = future_length
        self.obs_embedding_dim = obs_embedding_dim
        self.context_embedding_dim = context_embedding_dim

        self.obs_embedding = obs_embedding
        self.context_embedding = context_embedding
        self.future_obs_predictor = future_obs_predictor

    def forward(self, trajectories: torch.Tensor, prediction_offsets: torch.LongTensor):
        """
        Args:
            trajectories: (batch_size, seq_len, obs_dim)
            prediction_offsets: (batch_size,)
        """
        obs_embedding = apply_model_to_sequence(self.obs_embedding, trajectories)
        context_embedding = self.context_embedding(obs_embedding)
        future_obs_predictions = self.future_obs_predictor(context_embedding, prediction_offsets)
        return future_obs_predictions
    
    def embed_context(self):
        pass

def odyssey_cpc_model(
    context_length: int,
    future_length: int,
    obs_embedding_dim: int,
    context_embedding_dim: int,
    resnet_type: str = "resnet11",
    num_blocks: int = 4,
    hidden_size: int = 512,
    num_heads: int = 8,
) -> CPCModel:
    from odyssey.nn.nethack.tty_encoder import ResnetTTYEmbedding
    from odyssey.nn.contrastive.context_transformer import ContextTransformer
    from odyssey.nn.contrastive.linear_list import LinearList

    # Init obs embedding
    obs_embedding = ResnetTTYEmbedding(
        embedding_dim=obs_embedding_dim,
        char_embedding_dim=16,
        resnet_type="resnet11"
    )

    # Init context embedding
    context_embedding = ContextTransformer(
        obs_embedding_dim,
        context_embedding_dim,
        num_blocks=num_blocks,
        hidden_size=hidden_size,
        max_trajectory_length=context_length,
        num_heads=num_heads,
        drop_p=0.1
    )

    # Init future obs predictor
    future_obs_predictor = LinearList(
        context_embedding_dim,
        obs_embedding_dim,
        future_length
    )

    return CPCModel(
        context_length=context_length,
        future_length=future_length,
        obs_embedding_dim=obs_embedding_dim,
        context_embedding_dim=context_embedding_dim,
        obs_embedding=obs_embedding,
        context_embedding=context_embedding,
        future_obs_predictor=future_obs_predictor
    )