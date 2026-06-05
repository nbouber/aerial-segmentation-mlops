import torch
import hydra
import lightning as L

from lightning.pytorch.loggers import WandbLogger
from hydra.utils import instantiate


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg):
    data = instantiate(cfg.data)
    model = instantiate(cfg.model)

    wandb_logger = WandbLogger(project="Aerial Segmentation")
    trainer = L.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator="cuda" if torch.cuda.is_available() else "cpu",
        logger=wandb_logger,
        log_every_n_steps=cfg.training.log_every_n_steps,
    )
    trainer.fit(model, data)

if __name__ == "__main__":
    main()