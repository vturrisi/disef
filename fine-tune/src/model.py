import types
from collections import defaultdict
from typing import Any, List, Sequence, Tuple

import clip
import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningDataModule
from lightning.pytorch import LightningModule
from omegaconf import OmegaConf
from src.adaptation.lora import lora_replace_attention_layers
from src.data import templates
from src.utils.lr_scheduler import LinearWarmupCosineAnnealingLR
from src.utils.misc import accuracy_at_k, omegaconf_select, weighted_mean
from src.utils.mixup import Mixup
from timm.loss import SoftTargetCrossEntropy
from timm.models._manipulate import checkpoint_seq
from tqdm import tqdm


class Model(LightningModule):
    """LightningModule for DISEF.

    Args:
        cfg: config dict.
        clip (nn.Module): a pretrained CLIP model.
        datamodule (LightningDataModule): datamodule used for filtering classes.
        template (str): name of the dataset for retrieving its templates.
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

    def __init__(self, cfg, clip: nn.Module, datamodule: LightningDataModule, template: str):
        super().__init__()

        cfg = self.init_and_validate_cfg(cfg)

        self.cfg = cfg
        self.clip = clip

        self.base_new = False
        if isinstance(datamodule, list):
            datamodule_base, datamodule_new = datamodule
            self.base_new = True

        # this is (or can be) data related, so it's not always initialized with the model
        self.synthetic_data = omegaconf_select(cfg, "data.synthetic", False)

        # enable mixup/cutmix
        if cfg.mixing.enabled and (cfg.mixing.mixup > 0 or cfg.mixing.cutmix > 0):
            self.mixup_fn = Mixup(
                mixup_alpha=cfg.mixing.mixup,
                cutmix_alpha=cfg.mixing.cutmix,
                cutmix_minmax=None,
                prob=1.0,
                switch_prob=0.5,
                mode="batch",
                label_smoothing=cfg.mixing.label_smoothing,
                num_classes=len(datamodule_base.classes)
                if self.base_new
                else len(datamodule.classes),
            )
            # smoothing is handled with mixup label transform
            self.loss_fn = SoftTargetCrossEntropy()
        else:
            self.mixup_fn = None
            self.loss_fn = nn.CrossEntropyLoss(label_smoothing=cfg.mixing.label_smoothing)

        # visual model
        if cfg.lora.enabled:
            self.clip.visual.transformer = lora_replace_attention_layers(
                self.clip.visual.transformer,
                lora_r=cfg.lora.r,
                lora_alpha=cfg.lora.alpha,
                lora_dropout=cfg.lora.dropout,
                start_block=cfg.lora.start_block,
            )

        # text model
        if cfg.lora_llm.enabled:
            self.clip.transformer = lora_replace_attention_layers(
                self.clip.transformer,
                lora_r=cfg.lora_llm.r,
                lora_alpha=cfg.lora_llm.alpha,
                lora_dropout=cfg.lora_llm.dropout,
                start_block=cfg.lora_llm.start_block,
            )
        elif not cfg.freeze_llm:
            raise NotImplementedError("Only supports LoRA or freezing.")

        if self.base_new:
            self.register_buffer(
                "tokenized_text_base", self.tokenize_text(datamodule_base.classes, template)
            )
            self.register_buffer(
                "tokenized_text_new", self.tokenize_text(datamodule_new.classes, template)
            )

            # keep track of validation outputs to compute H
            self.validation_step_outputs = defaultdict(list)
        else:
            self.register_buffer("tokenized_text", self.tokenize_text(datamodule.classes, template))

        # enable checkpointing for text transformer
        # datasets with more classes simply go OOM if we don't do this
        def checkpoint_forward(self, x):
            x.requires_grad = True
            x = checkpoint_seq(self.resblocks, x)
            return x

        self.clip.transformer.forward = types.MethodType(checkpoint_forward, self.clip.transformer)

        # configure all learnable parameters
        self.set_learnable_params()

    def init_and_validate_cfg(self, cfg):
        # mixup/cutmix
        cfg.mixing = omegaconf_select(cfg, "mixing", {})
        cfg.mixing.enabled = omegaconf_select(cfg, "mixing.enabled", False)
        cfg.mixing.mixup = omegaconf_select(cfg, "mixing.mixup", 0.0)
        cfg.mixing.cutmix = omegaconf_select(cfg, "mixing.cutmix", 0.0)
        cfg.mixing.label_smoothing = omegaconf_select(cfg, "mixing.label_smoothing", 0.0)

        # visual model lora options
        cfg.lora = omegaconf_select(cfg, "lora", {})
        cfg.lora.enabled = omegaconf_select(cfg, "lora.enabled", False)
        if cfg.lora.enabled:
            cfg.lora.start_block = omegaconf_select(cfg, "lora.start_block", 0)
            assert not OmegaConf.is_missing(cfg, "lora.r")
            assert not OmegaConf.is_missing(cfg, "lora.alpha")
            assert not OmegaConf.is_missing(cfg, "lora.dropout")
        cfg.freeze_visual = omegaconf_select(cfg, "freeze_visual", not cfg.lora.enabled)

        # text model lora options
        cfg.lora_llm = omegaconf_select(cfg, "lora_llm", {})
        cfg.lora_llm.enabled = omegaconf_select(cfg, "lora_llm.enabled", False)
        if cfg.lora_llm.enabled:
            cfg.lora_llm.start_block = omegaconf_select(cfg, "lora_llm.start_block", 0)
            assert not OmegaConf.is_missing(cfg, "lora_llm.r")
            assert not OmegaConf.is_missing(cfg, "lora_llm.alpha")
            assert not OmegaConf.is_missing(cfg, "lora_llm.dropout")

        cfg.freeze_llm = omegaconf_select(cfg, "freeze_llm", not cfg.lora_llm.enabled)

        # weights
        cfg.weights = omegaconf_select(cfg, "weights", {})
        cfg.weights.real_ce = omegaconf_select(cfg, "weights.real_ce", 1.0)
        cfg.weights.synthetic_ce = omegaconf_select(cfg, "weights.synthetic_ce", 1.0)
        cfg.weights.z_loss = omegaconf_select(cfg, "weights.z_loss", 0.0)

        # optimizer options
        assert not OmegaConf.is_missing(cfg, "max_epochs")
        assert not OmegaConf.is_missing(cfg, "optim")
        assert not OmegaConf.is_missing(cfg, "optim.name")
        assert not OmegaConf.is_missing(cfg, "optim.lr")
        assert not OmegaConf.is_missing(cfg, "optim.weight_decay")
        cfg.optim.extra_args = omegaconf_select(cfg, "optim.extra_args", {})

        # scheduler options
        cfg.sched = omegaconf_select(cfg, "sched", {})
        cfg.sched.name = omegaconf_select(cfg, "sched.name", "none")
        if cfg.sched.name in ["warmup_cosine"]:
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

    @staticmethod
    def tokenize_text(classes, template):
        template = getattr(templates, f"{template}_template")

        texts = []
        for classname in tqdm(classes, desc="Tokenizing text"):
            class_texts = []
            for t in template:
                class_texts.append(t(classname))

            class_texts = clip.tokenize(class_texts)

            texts.append(class_texts)

        texts = torch.stack(texts)
        return texts

    def set_learnable_params(self):
        # turn off all parameters
        for p in self.parameters():
            p.requires_grad = False

        # learnable parameters for the visual model
        if self.cfg.lora.enabled:
            for name, p in self.clip.visual.named_parameters():
                if "lora_" in name:
                    p.requires_grad = True
        elif not self.cfg.freeze_visual:
            for p in self.clip.visual.parameters():
                p.requires_grad = True

        # learnable parameters for the text model
        if self.cfg.lora_llm.enabled:
            for name, p in self.clip.transformer.named_parameters():
                if "lora_" in name:
                    p.requires_grad = True

    @property
    def learnable_params(self):
        return [{"name": "all", "params": [p for p in self.parameters() if p.requires_grad]}]

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

    def forward_image(
        self,
        x: torch.Tensor,
    ):
        image_feats = self.clip.visual(x)
        image_feats = image_feats / image_feats.norm(dim=1, keepdim=True)
        return image_feats

    def forward_text(self, tokenized_text):
        n_classes, n_prompts = tokenized_text.shape[:2]
        tokenized_text = einops.rearrange(tokenized_text, "c p d -> (c p) d")
        with torch.set_grad_enabled(not self.cfg.freeze_llm):
            text_feats = self.clip.encode_text(tokenized_text)

        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

        # average across multiple prompt templates and re-norm
        text_feats = einops.rearrange(text_feats, "(c p) d -> c p d", c=n_classes, p=n_prompts)
        text_feats = text_feats.mean(dim=1)
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

        return text_feats

    def forward(
        self,
        x: torch.Tensor,
        tokenized_text: torch.Tensor,
        output_features: bool = False,
    ):
        image_feats = self.forward_image(x)
        text_feats = self.forward_text(tokenized_text)

        logit_scale = self.clip.logit_scale.exp()

        # no instance-specific text feats
        if len(text_feats.shape) == 2:
            # cosine similarity as logits
            logits_per_image = logit_scale * image_feats @ text_feats.t()
        else:
            logits_per_image = logit_scale * torch.stack(
                [image_feats[i] @ text_feats[i].t() for i in range(image_feats.shape[0])]
            )

        if output_features:
            return {
                "logits": logits_per_image,
                "image_feats": image_feats,
                "text_feats": text_feats,
            }

        return logits_per_image

    def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        # unpack batch
        if self.synthetic_data:
            (X, y), (X_synth, y_synth) = batch
        else:
            X, y = batch

        # mixup/cutmix data. latent features are not cutmixed
        if self.mixup_fn is not None:
            X, y = self.mixup_fn(X, y)
            if self.synthetic_data:
                X_synth, y_synth = self.mixup_fn(X_synth, y_synth)

        tokenized_text = self.tokenized_text_base if self.base_new else self.tokenized_text

        if self.synthetic_data:
            X_merged = torch.cat([X, X_synth])
            logits_merged = self(X_merged, tokenized_text)
            logits, logits_synth = torch.chunk(logits_merged, 2)

            ce_loss = self.loss_fn(logits, y)
            # auxiliary z-loss
            log_z = torch.logsumexp(logits, dim=-1)
            z_loss = (self.cfg.weights.z_loss * log_z**2).mean()

            ce_loss_synth = self.loss_fn(logits_synth, y_synth)
            loss = (
                ce_loss * self.cfg.weights.real_ce + ce_loss_synth * self.cfg.weights.synthetic_ce
            )

            metrics = {
                "train_loss": loss,
                "train_ce_loss": ce_loss,
                "train_ce_loss_synth": ce_loss_synth,
                "train_z_loss": z_loss,
            }
        else:
            logits = self(X, tokenized_text)
            ce_loss = self.loss_fn(logits, y)

            # auxiliary z-loss
            log_z = torch.logsumexp(logits, dim=-1)
            z_loss = (self.cfg.weights.z_loss * log_z**2).mean()

            loss = ce_loss

            metrics = {
                "train_loss": loss,
                "train_ce_loss": ce_loss,
                "train_z_loss": z_loss,
            }

        loss = loss + z_loss

        if self.mixup_fn is None:
            top_k_max = min(5, logits.size(1))
            acc1, acc5 = accuracy_at_k(logits, y, top_k=(1, top_k_max))
            metrics.update(
                {
                    "train_acc1": acc1,
                    "train_acc5": acc5,
                }
            )

        self.log_dict(metrics, on_epoch=True, sync_dist=True, batch_size=y.size(0))

        return loss

    def validation_step(
        self, batch: Sequence[Any], batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        X, y = batch

        # batch, n_diffusion_steps, dim
        if self.base_new:
            tokenized_text = (
                self.tokenized_text_base if dataloader_idx == 0 else self.tokenized_text_new
            )
        else:
            tokenized_text = self.tokenized_text

        logits = self(X, tokenized_text)

        loss = F.cross_entropy(logits, y)
        top_k_max = min(5, logits.size(1))
        acc1, acc5 = accuracy_at_k(logits, y, top_k=(1, top_k_max))

        metrics = {
            "val_loss": loss,
            "val_acc1": acc1,
            "val_acc5": acc5,
        }

        self.log_dict(metrics, sync_dist=True, batch_size=y.size(0))

        if self.base_new:
            self.validation_step_outputs[dataloader_idx].append(
                {**metrics, "batch_size": y.size(0)}
            )

        return loss

    def on_validation_epoch_end(self):
        if self.base_new:
            val_acc1_base = weighted_mean(self.validation_step_outputs[0], "val_acc1", "batch_size")
            val_acc1_new = weighted_mean(self.validation_step_outputs[1], "val_acc1", "batch_size")
            h = 2 / ((1 / val_acc1_base) + (1 / val_acc1_new))

            self.log_dict({"val_H": h}, sync_dist=True)

            self.validation_step_outputs.clear()
