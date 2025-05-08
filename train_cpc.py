from odyssey.nn.contrastive.context_transformer import ContextTransformer
from odyssey.nn.contrastive.linear_list import LinearList
from odyssey.nn.nethack.tty_embedding import TTYEmbeddingBase, ResnetTTYEmbedding
from odyssey.data.contrastive.cpc_dataset import CPCDataset

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict, tensorclass
from tensordict.nn import TensorDictModule
import lightning
from lightning.pytorch.loggers import WandbLogger
from torchmetrics.functional import accuracy

import hydra
from omegaconf import DictConfig

import argparse
from typing import List

@tensorclass
class TTYData:
    tty_chars: torch.LongTensor
    tty_colors: torch.LongTensor
    tty_cursor: torch.LongTensor

@tensorclass
class NethackCPCBatch:
    context: TTYData
    padding_mask: torch.BoolTensor
    positive_samples: TTYData
    positive_indices: torch.LongTensor

    @staticmethod
    def from_dict(dataset: CPCDataset, batch):
        return NethackCPCBatch(
            context=TTYData(**batch["context"], batch_size=(dataset.batch_size, dataset.context_length)),
            padding_mask=batch["padding_mask"],
            positive_samples=TTYData(**batch["positive_samples"]),
            positive_indices=batch["positive_indices"],
            batch_size=dataset.batch_size
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

        # self.save_hyperparameters()

    def forward(self, batch: NethackCPCBatch):
        positive_samples = self.embed_obs(batch.positive_samples)

        # The context is a sequence of observations
        # We first flatten it and after embedding them turn it back into sequences
        B, T, *S = batch.context.shape
        context = batch.context.reshape(-1, *S)
        context = self.embed_obs(context)
        context = context.reshape(B, T, -1)

        context_embedding = self.context_embedding(context, batch.padding_mask)
        
        positive_preds = self.future_obs_predictor(context_embedding, batch.positive_indices)
        return positive_samples, positive_preds
    
    def embed_obs(self, batch: TTYData):
        return self.tty_embedding(
            tty_chars=batch.tty_chars,
            tty_colors=batch.tty_colors,
            tty_cursor=batch.tty_cursor
        )

    def training_step(self, batch: NethackCPCBatch):
        positive_samples, positive_preds = self(batch)
        loss, acc = self.compute_loss_and_accuracy(positive_preds, positive_samples)

        self.log("loss", loss, prog_bar=True)
        self.log("accuracy", acc, prog_bar=True)

        return loss

    @torch.no_grad()
    def validation_step(self, batch: NethackCPCBatch):
        positive_samples, positive_preds = self(batch)
        loss, acc = self.compute_loss_and_accuracy(positive_preds, positive_samples)

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
        samples_per_trajectory=cfg.samples_per_trajectory,
        collate_fn=lambda x: NethackCPCBatch.from_dict(train_dataset, x)
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
        pin_memory=True,
        collate_fn=lambda x: NethackCPCBatch.from_dict(test_dataset, x)
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
