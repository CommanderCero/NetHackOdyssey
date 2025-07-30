from odyssey.nn.contrastive.context_transformer import ContextTransformer
from odyssey.nn.contrastive.linear_list import LinearList
from odyssey.nn.nethack.tty_encoder import TTYEncoderBase

import torch
import torch.nn.functional as F
from tensordict import tensorclass
import lightning
from torchmetrics.functional import accuracy

from typing import Optional, Callable

@tensorclass
class TTYData:
    tty_chars: torch.LongTensor
    tty_colors: torch.LongTensor
    tty_cursor: torch.LongTensor

    @classmethod
    def from_dict(cls, batch):
        return cls(
            tty_chars=batch["tty_chars"],
            tty_colors=batch["tty_colors"],
            tty_cursor=batch["tty_cursor"],
            batch_size=batch["tty_chars"].shape[:-2]
        )

@tensorclass
class NethackCPCBatch:
    context: TTYData
    padding_mask: torch.BoolTensor
    positive_samples: TTYData
    positive_indices: torch.LongTensor

    @classmethod
    def from_dict(cls, batch):
        return cls(
            context=TTYData.from_dict(batch["context"]),
            padding_mask=batch["padding_mask"],
            positive_samples=TTYData.from_dict(batch["positive_samples"]),
            positive_indices=batch["positive_indices"],
            batch_size=(batch["padding_mask"].shape[0],)
        )

class CPCModel(lightning.LightningModule):
    def __init__(self,
        tty_embedding: TTYEncoderBase,
        context_embedding: ContextTransformer,
        future_obs_predictor: LinearList,
        optimizer_fn: Callable[[], torch.optim.Optimizer],
        scheduler_fn: Optional[Callable[[], torch.optim.lr_scheduler.LRScheduler]] = None,
        lr: float=2e-4,
        compile: bool=False
    ):
        super().__init__()        
        
        self.save_hyperparameters(
            logger=False,
            ignore=["tty_embedding", "context_embedding", "future_obs_predictor"]
        )

        self.tty_embedding = tty_embedding
        self.context_embedding = context_embedding
        self.future_obs_predictor = future_obs_predictor

    def setup(self, stage: str):
        if self.hparams.compile and stage == "fit":
            self.tty_embedding = torch.compile(self.tty_embedding)
            self.context_embedding = torch.compile(self.context_embedding)
            self.future_obs_predictor = torch.compile(self.future_obs_predictor)

    def forward(self, batch):
        batch: NethackCPCBatch = NethackCPCBatch.from_dict(batch)
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

        # Debugging: log the model and batch if loss is NaN or too high
        if torch.isnan(loss) or loss >= 10:
            import os
            if "log_count" not in self.__dict__:
                self.log_count = 0
            self.log_count += 1
            log_dir = os.path.join(self.logger.save_dir, f"debug_epoch{self.current_epoch}_step{self.global_step}")
            os.makedirs(log_dir, exist_ok=True)

            if self.log_count < 5:
                torch.save(self.state_dict(), os.path.join(log_dir, "model.pt"))
                torch.save(batch, os.path.join(log_dir, "batch.pt"))

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
        optimizer = self.hparams.optimizer_fn(
            params=self.parameters(),
            lr=self.hparams.lr
        )
        if self.hparams.scheduler_fn is not None:
            scheduler = self.hparams.scheduler_fn(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }

        return optimizer
    
    def compute_loss_and_accuracy(self, queries: torch.Tensor, positive_keys: torch.Tensor, temperature: float = 0.1):
        queries = F.normalize(queries, dim=-1)
        positive_keys = F.normalize(positive_keys, dim=-1)

        logits = queries @ positive_keys.T
        labels = torch.arange(len(queries), device=queries.device)
        preds = torch.argmax(logits, dim=-1)

        loss = F.cross_entropy(
            logits / temperature,
            labels,
            reduction="mean"
        )
        acc = accuracy(preds, labels, task="multiclass", num_classes=len(queries))
        return loss, acc