from odyssey.nn.contrastive.context_transformer import ContextTransformer
from odyssey.nn.contrastive.linear_list import LinearList
from odyssey.nn.nethack.tty_embedding import TTYEmbeddingBase, ResnetTTYEmbedding
from odyssey.data.contrastive.cpc_dataset import CPCDataset, NethackCPCBatch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning
from lightning.pytorch.loggers import WandbLogger
from torchmetrics.functional import accuracy

import hydra
from omegaconf import DictConfig

import argparse
from typing import List

def info_nce_loss_accuracy(queries: torch.Tensor, positive_keys: torch.Tensor, temperature: float = 0.1, reduction="mean"):
    """
    Computes the InfoNCE loss.
    Args:
        query: The query tensor of shape (B, D).
        positive_keys: The positive keys tensor of shape (B, D).
        temperature: The temperature for scaling the logits.
        reduction: The reduction method to apply to the loss.
    Returns:
        The InfoNCE loss.
    """
    queries = F.normalize(queries, dim=-1)
    positive_keys = F.normalize(positive_keys, dim=-1)

    logits = queries @ positive_keys.T
    # The positive logits are the diagonal of the logits matrix
    # Everything else is a negative sample
    labels = torch.arange(len(queries), device=queries.device)
    preds = torch.argmax(logits, dim=-1)

    loss = F.cross_entropy(
        logits / temperature,
        labels,
        reduction=reduction
    )

    acc = accuracy(preds, labels, task="multiclass", num_classes=len(queries))

    return loss, acc

class InfoNCELossAccuracy(nn.Module):
    def __init__(self, query_key, positive_sample_key, temperature=0.1, reduction='mean'):
        super().__init__()
        self.query_key = query_key
        self.positive_sample_key = positive_sample_key
        self.temperature = temperature
        self.reduction = reduction

    def forward(self, batch: TensorDict):
        return info_nce_loss_accuracy(
            batch[self.query_key],
            batch[self.positive_sample_key],
            temperature=self.temperature,
            reduction=self.reduction
        )
    

class CPCModel(lightning.LightningModule):
    def __init__(self,
        tty_embedding: TTYEmbeddingBase,
        context_embedding: ContextTransformer,
        future_obs_predictor: LinearList
    ):
        super().__init__()
        self.tty_embedding = tty_embedding
        self.context_embedding = context_embedding
        self.future_obs_predictor = future_obs_predictor

        self.save_hyperparameters()

    def forward(self, batch: NethackCPCBatch):
        batch["positive_samples"] = self.obs_embedding(batch["positive_samples"])
        batch["context"] = self.embed_context_obs(batch["context"])
        batch = self.context_embedding(batch)
        batch = self.future_obs_predictor(batch)

        return batch
    
    def embed_context_obs(self, context):
        B, T, *C = context.shape
        context = context.reshape(-1, *C)
        context = self.obs_embedding(context)
        context = context.reshape(B, T, *C)
        return context

    def training_step(self, batch: NethackCPCBatch):
        batch = self(batch)
        loss, acc = self.loss_accuracy_fn(batch)

        self.log("loss", loss, prog_bar=True)
        self.log("accuracy", acc, prog_bar=True)

        return loss

    @torch.no_grad()
    def validation_step(self, batch: NethackCPCBatch):
        batch = self(batch)

        self.log("val_loss", loss)
        self.log("val_accuracy", acc)

        return {"val_loss": loss, "val_accuracy": acc}
    
    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=2e-4,
        )
    
    def compute_loss_and_accuracy(self, queries: torch.Tensor, positive_keys: torch.Tensor):
        queries = F.normalize(queries, dim=-1)
        positive_keys = F.normalize(positive_keys, dim=-1)

        logits = queries @ positive_keys.T
        labels = torch.arange(len(queries), device=queries.device)
        preds = torch.argmax(logits, dim=-1)

        loss = F.cross_entropy(
            logits / self.temperature,
            labels,
            reduction=self.reduction
        )
        acc = accuracy(preds, labels, task="multiclass", num_classes=len(queries))
        return loss, acc

@hydra.main(config_path="config", config_name="train_cpc_config", version_base="1.3")
def main(cfg: DictConfig):
    # Init tty_embedding
    tty_embedding = ResnetTTYEmbedding(
        embedding_dim=cfg.obs_embedding_dim,
        char_embedding_dim=cfg.char_embedding_dim,
        resnet_type="resnet11"
    )

    # Init context embedding
    context_embedding = ContextTransformer(
        cfg.obs_embedding_dim,
        cfg.context_embedding_dim,
        num_blocks=4,
        hidden_size=512,
        max_trajectory_length=cfg.context_length,
        num_heads=8,
        drop_p=0.1
    )

    # Init future obs predictor
    future_obs_predictor = LinearList(
        cfg.context_embedding_dim,
        cfg.obs_embedding_dim,
        cfg.future_length,
        bias=False
    )

    # Datasets
    train_dataset = CPCDataset(
        cfg.train_data_path,
        batch_size=cfg.batch_size,
        context_length=cfg.context_length,
        future_length=cfg.future_length,
        samples_per_trajectory=cfg.samples_per_trajectory
    )

    test_dataset = CPCDataset(
        cfg.test_data_path,
        batch_size=cfg.batch_size,
        context_length=cfg.context_length,
        future_length=cfg.future_length,
        samples_per_trajectory=cfg.samples_per_trajectory
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=None,
        num_workers=2,
        pin_memory=True
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=None,
        num_workers=2,
        pin_memory=True
    )

    # Loss
    loss_accuracy_fn = InfoNCELossAccuracy(
        query_key="obs_preds",
        positive_sample_key=("positive_samples", "obs_embedding"),
    )

    # Model
    model = CPCModel(
        tty_embedding=tty_embedding,
        context_embedding=context_embedding,
        future_obs_predictor=future_obs_predictor
    )

    logger = WandbLogger(
        name="train_cpc",
        project="NethackOdyssey",
        log_model="all",
    )

    trainer = lightning.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=cfg.max_epochs,
        limit_train_batches=cfg.steps_per_epoch,
        limit_val_batches=cfg.evaluation_steps,
        logger=logger,
        log_every_n_steps=50,
    )

    # Train
    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=test_loader,
    )

if __name__ == "__main__":
    main()
