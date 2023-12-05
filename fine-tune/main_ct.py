import inspect
from argparse import ArgumentParser

import clip
import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers.wandb import WandbLogger
from omegaconf import OmegaConf
from src.data import DATA
from src.model_ct import Model
from src.utils.misc import omegaconf_select
from src.zero_shot_classifier import build_zero_shot_classifier
from torchinfo import summary


def main():
    parser = ArgumentParser()
    parser.add_argument("--cfg", required=True)
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Any key=value arguments to override config values (use dots for.nested=overrides)",
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.cfg)
    OmegaConf.set_struct(cfg, False)

    new_cfg = OmegaConf.from_dotlist(args.overrides)
    cfg = OmegaConf.merge(cfg, new_cfg)

    seed = omegaconf_select(cfg, "seed", 5)
    seed_everything(seed)

    datamodule = DATA[cfg.data.name.lower()](
        k_shot=cfg.data.k_shot,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        synthetic=omegaconf_select(cfg.data, "synthetic", False),
        synthetic_data_dir=omegaconf_select(cfg.data, "synthetic_data_dir", None),
        maximum_synthetic_samples=omegaconf_select(cfg.data, "maximum_synthetic_samples", -1),
        augment=omegaconf_select(cfg.data, "augment", True),
        subsample=omegaconf_select(cfg.data, "subsample", "all"),
        seed=seed,
        classifier_tuning=True,  # used for data augmentations
    )

    clip_model, _ = clip.load(cfg.model.name, device="cpu")
    zero_shot_classifier = build_zero_shot_classifier(
        clip_model, datamodule, template=cfg.data.name.lower()
    )
    model = Model(cfg, clip_model.visual, zero_shot_classifier)

    pretrained_model = omegaconf_select(cfg, "pretrained_model", None)
    if pretrained_model:
        checkpoint = torch.load(pretrained_model, map_location="cpu")["state_dict"]
        model.load_state_dict(checkpoint)

    callbacks = []
    if cfg.wandb.enabled:
        wandb_logger = WandbLogger(
            name=cfg.name,
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            offline=cfg.wandb.offline,
        )
        wandb_logger.watch(model, log="gradients", log_freq=100)
        wandb_logger.log_hyperparams(OmegaConf.to_container(cfg))

        # lr logging
        lr_monitor = LearningRateMonitor(logging_interval="step")
        callbacks.append(lr_monitor)

    trainer_kwargs = OmegaConf.to_container(cfg)
    valid_kwargs = inspect.signature(Trainer.__init__).parameters
    trainer_kwargs = {name: trainer_kwargs[name] for name in valid_kwargs if name in trainer_kwargs}
    trainer_kwargs.update(
        {
            "logger": wandb_logger if cfg.wandb.enabled else None,
            "callbacks": callbacks,
            "enable_checkpointing": omegaconf_select(cfg, "save_checkpoint", False),
            "log_every_n_steps": 10,
        }
    )

    summary(model, (cfg.data.batch_size, 3, 224, 224), device="cpu", depth=4)

    trainer = Trainer(**trainer_kwargs)
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
