import torch
import hydra
import lightning as L

from lightning.pytorch.loggers import WandbLogger
from hydra.utils import instantiate
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from hydra.core.hydra_config import HydraConfig


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg):
    data = instantiate(cfg.data)
    model = instantiate(cfg.model)

    checkpoint_callback = ModelCheckpoint(
        dirpath=HydraConfig.get().runtime.output_dir + "/" + cfg.training.checkpoint_dir,
        filename="best-checkpoint",
        save_top_k=1,
        monitor=cfg.training.monitor,
        mode="max"
    )
    early_stopping_callback = EarlyStopping(
        monitor=cfg.training.monitor,
        patience=cfg.training.patience,
        mode="max"
    )

    wandb_logger = WandbLogger(project="Aerial Segmentation")
    trainer = L.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator="cuda" if torch.cuda.is_available() else "cpu",
        logger=wandb_logger,
        log_every_n_steps=cfg.training.log_every_n_steps,
        callbacks=[checkpoint_callback, early_stopping_callback]
    )
    trainer.fit(model, data)

if __name__ == "__main__":
    main()