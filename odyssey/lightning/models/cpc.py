from odyssey.nn.contrastive.context_transformer import ContextTransformer
from odyssey.nn.contrastive.linear_list import LinearList
from odyssey.nn.nethack.tty_embedding import TTYEmbeddingBase

import torch
import torch.nn.functional as F
from tensordict import tensorclass
import lightning
from torchmetrics.functional import accuracy

import hydra
from omegaconf import DictConfig

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
        tty_embedding: TTYEmbeddingBase,
        context_embedding: ContextTransformer,
        future_obs_predictor: LinearList
    ):
        super().__init__()
        self.tty_embedding = tty_embedding
        self.context_embedding = context_embedding
        self.future_obs_predictor = future_obs_predictor

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