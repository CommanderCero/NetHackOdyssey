from odyssey.lightning.models.cpc import CPCModel
from odyssey.lightning.data.cpc_module import CPCDataModule

import torch
import lightning
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import Logger

import hydra
from omegaconf import DictConfig
from typing import List, Dict

@hydra.main(config_path="config", config_name="train_cpc", version_base="1.3")
def main(cfg: DictConfig):
    if cfg.float32_matmul_precision:
        torch.set_float32_matmul_precision(cfg.float32_matmul_precision)

    datamodule: CPCDataModule = hydra.utils.instantiate(cfg.data)
    if cfg.prepare_data_only:
        print("prepare_data_only flag is set to True. Only preparing data...")
        datamodule.prepare_data()
        return

    model: CPCModel = hydra.utils.instantiate(cfg.model)
    logger: Logger = hydra.utils.instantiate(cfg.logger)
    callbacks: Dict[str, Callback] = hydra.utils.instantiate(cfg.callbacks)
    trainer: lightning.Trainer = hydra.utils.instantiate(
        cfg.trainer,
        logger=logger,
        callbacks=list(callbacks.values())
    )

    # Train
    trainer.fit(
        model=model,
        datamodule=datamodule,
    )

if __name__ == "__main__":
    main()
