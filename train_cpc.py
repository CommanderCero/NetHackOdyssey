from odyssey.nn.contrastive.context_transformer import ContextTransformer
from odyssey.nn.contrastive.linear_list import LinearList
from odyssey.nn.nethack.tty_embedding import ResnetTTYEmbedding
from odyssey.data.contrastive.cpc_dataset import CPCDataset
from tensordict import TensorDict

import torch
import torch.nn as nn
import lightning

class CPCModel(lightning.LightningModule):
    def __init__(self,
        obs_embedding: nn.Module,
        context_embedding: nn.Module,
        future_obs_predictor: nn.Module,
    ):
        super().__init__()
        self.obs_embedding = obs_embedding
        self.context_embedding = context_embedding
        self.future_obs_predictor = future_obs_predictor

    def training_step(self, batch):
        obs = self.obs_embedding(**batch["context"])
        return None
    
    def embed_context(self, context_batch):
        pass
    
    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=2e-4,
        )

if __name__ == "__main__":
    char_embedding_dim = 16
    obs_embedding_dim = 256
    context_embedding_dim = 512
    context_length = 100
    future_length = 30

    # Init obs embedding
    obs_embedding = ResnetTTYEmbedding(
        embedding_dim=obs_embedding_dim,
        char_embedding_dim=char_embedding_dim,
        resnet_type="resnet11"
    )

    # Init context embedding
    context_embedding = ContextTransformer(
        obs_embedding_dim,
        context_embedding_dim,
        num_blocks=4,
        hidden_size=512,
        max_trajectory_length=context_length,
        num_heads=8,
        drop_p=0.1
    )

    # Init future obs predictor
    # Holy f is this one big, find a better way to do this that requires much less parameters
    future_obs_predictor = LinearList(
        context_embedding_dim,
        obs_embedding_dim,
        future_length,
        bias=False
    )
    
    # Dataloader
    dataset = CPCDataset(
        "/workspace/data/nld_nao.h5",
        batch_size=32,
        context_length=100,
        future_length=30,
        samples_per_trajectory=4
    )

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=None,
        num_workers=2,
        pin_memory=True,
        collate_fn=lambda batch: TensorDict.from_dict(batch, auto_batch_size=True)
    )

    # Setup for training
    model = CPCModel(
        obs_embedding=obs_embedding,
        context_embedding=context_embedding,
        future_obs_predictor=future_obs_predictor
    )

    trainer = lightning.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=1,
    )
    
    # Train the model
    trainer.fit(
        model=model, 
        train_dataloaders=dataloader
    )