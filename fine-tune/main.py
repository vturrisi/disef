import inspect
from argparse import ArgumentParser

import clip
import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.strategies import DDPStrategy
from omegaconf import OmegaConf
from src.data import DATA
from src.model import Model
from src.model_learnable_prompts import Model as ModelLearnablePrompts
from src.utils.misc import omegaconf_select
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

    # load generic config
    cfg = OmegaConf.load(args.cfg)
    OmegaConf.set_struct(cfg, False)

    # merged it with bash arguments, giving preference to those
    new_cfg = OmegaConf.from_dotlist(args.overrides)
    cfg = OmegaConf.merge(cfg, new_cfg)

    seed = omegaconf_select(cfg, "seed", 5)
    seed_everything(seed)

    # select the correct model to use
    if omegaconf_select(cfg, "visual_prompts.enabled", False) or omegaconf_select(
        cfg, "text_prompts.enabled", False
    ):
        ModelClass = ModelLearnablePrompts
    elif omegaconf_select(cfg, "lora.enabled", False) or omegaconf_select(
        cfg, "lora_llm.enabled", False
    ):
        ModelClass = Model
    else:
        raise NotImplementedError

    # load data for either all classes (default scenario) or base/new scenario
    base_new = omegaconf_select(cfg.data, "base_new", False)
    if base_new:
        datamodule_base = DATA[cfg.data.name.lower()](
            k_shot=cfg.data.k_shot,
            batch_size=cfg.data.batch_size,
            val_batch_size=omegaconf_select(cfg.data, "val_batch_size", cfg.data.batch_size),
            num_workers=cfg.data.num_workers,
            synthetic=omegaconf_select(cfg.data, "synthetic", False),
            synthetic_data_dir=omegaconf_select(cfg.data, "synthetic_data_dir", None),
            maximum_synthetic_samples=omegaconf_select(cfg.data, "maximum_synthetic_samples", -1),
            augment=omegaconf_select(cfg.data, "augment", True),
            subsample="base",
            randaugment_n=omegaconf_select(cfg.data, "randaugment_n", 2),
            randaugment_m=omegaconf_select(cfg.data, "randaugment_m", 9),
            concat_mode=omegaconf_select(cfg.data, "concat_mode", "max"),
            seed=seed,
        )
        datamodule_new = DATA[cfg.data.name.lower()](
            k_shot=cfg.data.k_shot,
            batch_size=cfg.data.batch_size,
            val_batch_size=omegaconf_select(cfg.data, "val_batch_size", cfg.data.batch_size),
            num_workers=cfg.data.num_workers,
            synthetic=omegaconf_select(cfg.data, "synthetic", False),
            synthetic_data_dir=omegaconf_select(cfg.data, "synthetic_data_dir", None),
            maximum_synthetic_samples=omegaconf_select(cfg.data, "maximum_synthetic_samples", -1),
            augment=omegaconf_select(cfg.data, "augment", True),
            subsample="new",
            randaugment_n=omegaconf_select(cfg.data, "randaugment_n", 2),
            randaugment_m=omegaconf_select(cfg.data, "randaugment_m", 9),
            seed=seed,
        )
        print(f"Base classes: {len(datamodule_base.classes)}", datamodule_base.classes)
        print(f"New classes: {len(datamodule_new.classes)}", datamodule_new.classes)

        clip_model, _ = clip.load(cfg.model.name, device="cpu")
        model = ModelClass(
            cfg,
            clip_model,
            [datamodule_base, datamodule_new],
            template=cfg.data.name.lower(),
        )

        datamodule_base.setup()
        datamodule_new.setup()

        datamodule = datamodule_base
        datamodule.data_val = [datamodule_base.data_val, datamodule_new.data_val]
        datamodule.data_test = datamodule.data_val

    else:
        datamodule = DATA[cfg.data.name.lower()](
            k_shot=cfg.data.k_shot,
            batch_size=cfg.data.batch_size,
            val_batch_size=omegaconf_select(cfg.data, "val_batch_size", cfg.data.batch_size),
            num_workers=cfg.data.num_workers,
            synthetic=omegaconf_select(cfg.data, "synthetic", False),
            synthetic_data_dir=omegaconf_select(cfg.data, "synthetic_data_dir", None),
            maximum_synthetic_samples=omegaconf_select(cfg.data, "maximum_synthetic_samples", -1),
            augment=omegaconf_select(cfg.data, "augment", True),
            randaugment_n=omegaconf_select(cfg.data, "randaugment_n", 2),
            randaugment_m=omegaconf_select(cfg.data, "randaugment_m", 9),
            concat_mode=omegaconf_select(cfg.data, "concat_mode", "max"),
            seed=seed,
        )
        print(f"Total classes: {len(datamodule.classes)}", datamodule.classes)

        clip_model, _ = clip.load(cfg.model.name, device="cpu")
        model = ModelClass(cfg, clip_model, datamodule, template=cfg.data.name.lower())

    # load wandb callbacks
    callbacks = []
    if cfg.wandb.enabled:
        wandb_logger = WandbLogger(
            name=cfg.name,
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            offline=cfg.wandb.offline,
        )
        wandb_logger.watch(model, log="gradients")
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
    if "ddp" in trainer_kwargs["strategy"]:
        trainer_kwargs["strategy"] = DDPStrategy(static_graph=True)

    summary(
        model,
        input_data=[
            torch.randn((cfg.data.batch_size, 3, 224, 224)),  # emulate random image
            torch.randint(0, 100, (10, 1, 77)),  # emulate random text
        ],
        device="cpu",
        depth=4,
    )

    trainer = Trainer(**trainer_kwargs)
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
