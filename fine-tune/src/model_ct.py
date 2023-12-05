from typing import Any, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch import LightningModule
from omegaconf import OmegaConf
from src.utils.lr_scheduler import LinearWarmupCosineAnnealingLR
from src.utils.misc import accuracy_at_k, omegaconf_select


class Model(LightningModule):
    """LightningModule for Classifier Tuning.

    Args:
        cfg: config dict.
        backbone (nn.Module): CLIP's vision backbone.
        classifier (nn.Module): linear classifier constructed from CLIP's zero-shot classifier.
    """

    _OPTIMIZERS = {
        "sgd": torch.optim.SGD,
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
    }
    _SCHEDULERS = [
        "warmup_cosine",
        "none",
    ]

    def __init__(self, cfg, backbone: nn.Module, classifier: nn.Module):
        super().__init__()

        cfg = self.init_and_validate_cfg(cfg)

        self.cfg = cfg
        self.backbone = backbone
        self.classifier = classifier

        # set training options for the backbone
        for p in self.backbone.parameters():
            p.requires_grad = False

        # start from a randomly init classifier
        if cfg.reset_classifier:
            self.classifier = nn.Linear(self.classifier.in_features, self.classifier.out_features)

        self.synthetic_data = omegaconf_select(cfg, "data.synthetic", False)

    def init_and_validate_cfg(self, cfg):
        # general options
        assert not OmegaConf.is_missing(cfg, "max_epochs")

        # train model options
        cfg.freeze_backbone = omegaconf_select(cfg, "freeze_backbone", True)
        cfg.reset_classifier = omegaconf_select(cfg, "reset_classifier", False)

        # weights
        cfg.weights = omegaconf_select(cfg, "weights", {})
        cfg.weights.real_ce = omegaconf_select(cfg, "weights.real_ce", 1.0)
        cfg.weights.synthetic_ce = omegaconf_select(cfg, "weights.synthetic_ce", 1.0)
        cfg.weights.z_loss = omegaconf_select(cfg, "weights.z_loss", 0.0)

        # optimizer options
        assert not OmegaConf.is_missing(cfg, "optim")
        assert not OmegaConf.is_missing(cfg, "optim.name")
        assert not OmegaConf.is_missing(cfg, "optim.lr")
        assert not OmegaConf.is_missing(cfg, "optim.weight_decay")
        cfg.optim.classifier_lr = omegaconf_select(cfg, "optim.classifier_lr", -1)
        cfg.optim.extra_args = omegaconf_select(cfg, "optim.extra_args", {})

        # scheduler options
        cfg.sched = omegaconf_select(cfg, "sched", {})
        cfg.sched.name = omegaconf_select(cfg, "sched.name", "none")
        if cfg.sched.name == "warmup_cosine":
            cfg.sched.warmup_epochs = omegaconf_select(cfg, "sched.warmup_epochs", 0)
            cfg.sched.warmup_start_lr = omegaconf_select(cfg, "sched.warmup_start_lr", 0)
            cfg.sched.min_lr = omegaconf_select(cfg, "sched.min_lr", 0)
            cfg.sched.interval = omegaconf_select(cfg, "sched.interval", "step")

        # wandb options
        cfg.wandb = omegaconf_select(cfg, "wandb", {})
        cfg.wandb.enabled = omegaconf_select(cfg, "wandb.enabled", False)
        cfg.wandb.project = omegaconf_select(cfg, "wandb.project", "zero-shot")
        cfg.wandb.entity = omegaconf_select(cfg, "wandb.entity", None)
        cfg.wandb.offline = omegaconf_select(cfg, "wandb.offline", False)

        return cfg

    @property
    def learnable_params(self):
        return [p for p in self.parameters() if p.requires_grad]

    def configure_optimizers(self) -> Tuple[List, List]:
        """Collects learnable parameters and configures the optimizer and learning rate scheduler.

        Returns:
            Tuple[List, List]: two lists containing the optimizer and the scheduler.
        """

        learnable_params = self.learnable_params

        # create optimizer
        optimizer = self._OPTIMIZERS[self.cfg.optim.name]
        optimizer = optimizer(
            learnable_params,
            lr=self.cfg.optim.lr,
            weight_decay=self.cfg.optim.weight_decay,
            **self.cfg.optim.extra_args,
        )

        scheduler_name = self.cfg.sched.name.lower()
        if scheduler_name == "none":
            return optimizer
        elif scheduler_name == "warmup_cosine":
            max_warmup_steps = (
                self.cfg.sched.warmup_epochs
                * (self.trainer.estimated_stepping_batches / self.cfg.max_epochs)
                if self.cfg.sched.interval == "step"
                else self.cfg.sched.warmup_epochs
            )
            max_scheduler_steps = (
                self.trainer.estimated_stepping_batches
                if self.cfg.sched.interval == "step"
                else self.cfg.max_epochs
            )
            scheduler = {
                "scheduler": LinearWarmupCosineAnnealingLR(
                    optimizer,
                    warmup_epochs=max_warmup_steps,
                    max_epochs=max_scheduler_steps,
                    warmup_start_lr=self.cfg.sched.warmup_start_lr
                    if self.cfg.sched.warmup_epochs > 0
                    else self.cfg.optim.lr,
                    eta_min=self.cfg.sched.min_lr,
                ),
                "interval": self.cfg.sched.interval,
                "frequency": 1,
            }
        else:
            raise ValueError(f"{scheduler_name} not in {self._SCHEDULERS}")

        return [optimizer], [scheduler]

    def forward(self, x: torch.Tensor):
        return self.classifier(self.backbone(x))

    def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        if self.synthetic_data:
            (X, y), (X_synth, y_synth) = batch

            # real data
            logits = self(X)
            loss = F.cross_entropy(logits, y)

            # synthetic data
            logits_synth = self(X_synth)
            loss_synth = F.cross_entropy(logits_synth, y_synth)

            loss = loss * self.cfg.weights.real_ce + loss_synth * self.cfg.weights.synthetic_ce
        else:
            X, y = batch

            logits = self(X)
            loss = F.cross_entropy(logits, y)

        batch_size = y.size(0)
        top_k_max = min(5, logits.size(1))
        acc1, acc5 = accuracy_at_k(logits, y, top_k=(1, top_k_max))
        self.log_dict(
            {"train_loss": loss, "train_acc1": acc1, "train_acc5": acc5},
            on_epoch=True,
            sync_dist=True,
            batch_size=batch_size,
        )

        return loss

    def validation_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        X, y = batch

        logits = self(X)

        batch_size = y.size(0)
        loss = F.cross_entropy(logits, y)
        top_k_max = min(5, logits.size(1))
        acc1, acc5 = accuracy_at_k(logits, y, top_k=(1, top_k_max))

        metrics = {
            "val_loss": loss,
            "val_acc1": acc1,
            "val_acc5": acc5,
        }

        self.log_dict(metrics, sync_dist=True, batch_size=batch_size)

        return loss
