import math
import types
from collections import defaultdict
from functools import reduce
from operator import mul
from typing import Any, List, Sequence, Tuple

import clip
import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningDataModule
from lightning.pytorch import LightningModule
from omegaconf import OmegaConf
from src.data import templates
from src.utils.lr_scheduler import LinearWarmupCosineAnnealingLR
from src.utils.misc import accuracy_at_k, omegaconf_select, weighted_mean
from src.utils.mixup import Mixup
from timm.loss import SoftTargetCrossEntropy
from timm.models._manipulate import checkpoint_seq
from tqdm import tqdm


class Model(LightningModule):
    """LightningModule for VPT, TPT and VPT + TPT.

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

        assert (
            cfg.visual_prompts.enabled or cfg.text_prompts.enabled
        ), "Either visual prompts or text prompts should be enabled"

        if cfg.visual_prompts.enabled:
            self.create_visual_prompts()

        if cfg.text_prompts.enabled:
            if self.base_new:
                self.create_text_base_new_prompts(datamodule_base.classes, datamodule_new.classes)
            else:
                self.create_text_prompts(datamodule.classes)
        else:
            if self.base_new:
                self.register_buffer(
                    "tokenized_text_base", self.tokenize_text(datamodule_base.classes, template)
                )
                self.register_buffer(
                    "tokenized_text_new", self.tokenize_text(datamodule_new.classes, template)
                )
            else:
                self.register_buffer(
                    "tokenized_text", self.tokenize_text(datamodule.classes, template)
                )

        if self.base_new:
            # keep track of validation outputs to compute H
            self.validation_step_outputs = defaultdict(list)

        # enable checkpointing for text transformer
        # datasets with more classes simply go OOM if we don't do this
        if not cfg.text_prompts.enabled:

            def checkpoint_forward(self, x):
                x.requires_grad = True
                x = checkpoint_seq(self.resblocks, x)
                return x

            self.clip.transformer.forward = types.MethodType(
                checkpoint_forward, self.clip.transformer
            )

        # configure all learnable parameters
        self.set_learnable_params()

    def init_and_validate_cfg(self, cfg):
        # mixup/cutmix
        cfg.mixing = omegaconf_select(cfg, "mixing", {})
        cfg.mixing.enabled = omegaconf_select(cfg, "mixing.enabled", False)
        cfg.mixing.mixup = omegaconf_select(cfg, "mixing.mixup", 0.0)
        cfg.mixing.cutmix = omegaconf_select(cfg, "mixing.cutmix", 0.0)
        cfg.mixing.label_smoothing = omegaconf_select(cfg, "mixing.label_smoothing", 0.0)

        # visual prompt tuning (shallow)
        cfg.visual_prompts = omegaconf_select(cfg, "visual_prompts", {})
        cfg.visual_prompts.enabled = bool(omegaconf_select(cfg, "visual_prompts.enabled", False))
        if cfg.visual_prompts.enabled:
            assert not OmegaConf.is_missing(cfg, "visual_prompts.number")

        # text prompt tuning (shallow)
        cfg.text_prompts = omegaconf_select(cfg, "text_prompts", {})
        cfg.text_prompts.enabled = bool(omegaconf_select(cfg, "text_prompts.enabled", False))
        if cfg.text_prompts.enabled:
            assert not OmegaConf.is_missing(cfg, "text_prompts.number")

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

    def create_visual_prompts(self):
        num_prompts = self.cfg.visual_prompts.number
        patch_size = self.clip.visual.conv1.kernel_size
        prompt_dim = self.clip.visual.conv1.out_channels

        val = math.sqrt(6.0 / float(3 * reduce(mul, patch_size, 1) + prompt_dim))  # noqa
        self.clip.visual.prompt_embeddings = nn.Parameter(torch.zeros(1, num_prompts, prompt_dim))
        nn.init.uniform_(self.clip.visual.prompt_embeddings.data, -val, val)

        def forward(self, x: torch.Tensor):
            x = self.conv1(x)  # shape = [*, width, grid, grid]
            x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
            x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
            x = torch.cat(
                [
                    self.class_embedding.to(x.dtype)
                    + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
                    x,
                ],
                dim=1,
            )  # shape = [*, grid ** 2 + 1, width]
            x = x + self.positional_embedding.to(x.dtype)

            # add learnable prompts
            B = x.shape[0]
            prompts = self.prompt_embeddings.expand(B, -1, -1)

            # CLS token, learnable prompts, image patches
            x = torch.cat((x[:, :1, :], prompts, x[:, 1:, :]), dim=1)
            x = self.ln_pre(x)

            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.transformer(x)
            x = x.permute(1, 0, 2)  # LND -> NLD

            x = self.ln_post(x[:, 0, :])

            if self.proj is not None:
                x = x @ self.proj

            return x

        self.clip.visual.forward = types.MethodType(forward, self.clip.visual)

    def create_text_prompts(self, classes):
        num_prompts = self.cfg.text_prompts.number
        prompt_dim = self.clip.transformer.width

        # using a single prompt for all classes
        prompt_embeddings = torch.zeros(1, num_prompts, prompt_dim)
        # normal initialization following COOP
        nn.init.normal_(prompt_embeddings, std=0.02)
        # repeat the same embeddings for all classes
        prompt_embeddings = prompt_embeddings.repeat(len(classes), 1, 1)
        self.clip.prompt_embeddings = nn.Parameter(prompt_embeddings)

        # create "textual" version of the prompts
        prompt_prefix = " ".join(["X"] * num_prompts)
        text = [prompt_prefix + " " + name + "." for name in classes]
        text = clip.tokenize(text)
        # we use this to know which embeddings to use as output of the LLM
        self.clip.tokenized_text = text

        # embed text to extract SOS, CLS and EOS tokens (that are not learnabled)
        with torch.no_grad():
            embedding = self.clip.token_embedding(text)
        self.clip.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.clip.register_buffer("token_suffix", embedding[:, 1 + num_prompts :, :])  # CLS, EOS

        def encode_text(self):
            prompts = self.prompt_embeddings
            x = torch.cat([self.token_prefix, prompts, self.token_suffix], dim=1)

            x = x + self.positional_embedding.type(self.dtype)
            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.transformer(x)
            x = x.permute(1, 0, 2)  # LND -> NLD
            x = self.ln_final(x).type(self.dtype)

            # x.shape = [batch_size, n_ctx, transformer.width]
            # take features from the eot embedding (eot_token is the highest
            # number in each sequence)
            x = (
                x[torch.arange(x.shape[0]), self.tokenized_text.argmax(dim=-1)]
                @ self.text_projection
            )

            return x

        self.clip.encode_text = types.MethodType(encode_text, self.clip)

    def create_text_base_new_prompts(self, base_classes, new_classes):
        num_prompts = self.cfg.text_prompts.number
        prompt_dim = self.clip.transformer.width

        # using a single prompt for all classes
        prompt_embeddings = torch.zeros(1, num_prompts, prompt_dim)
        # normal initialization following COOP
        nn.init.normal_(prompt_embeddings, std=0.02)
        # repeat the same embeddings for all classes
        self.clip.prompt_embeddings = nn.Parameter(prompt_embeddings)

        # create "textual" version of the base prompts
        prompt_prefix = " ".join(["X"] * num_prompts)
        text = [prompt_prefix + " " + name + "." for name in base_classes]
        text = clip.tokenize(text)
        # we use this to know which embeddings to use as output of the LLM
        self.clip.tokenized_text_base = text

        # embed text to extract SOS, CLS and EOS tokens (that are not learnabled)
        with torch.no_grad():
            embedding = self.clip.token_embedding(text)
        self.clip.register_buffer("token_prefix_base", embedding[:, :1, :])  # SOS
        self.clip.register_buffer(
            "token_suffix_base", embedding[:, 1 + num_prompts :, :]
        )  # CLS, EOS

        # create "textual" version of the new prompts
        prompt_prefix = " ".join(["X"] * num_prompts)
        text = [prompt_prefix + " " + name + "." for name in new_classes]
        text = clip.tokenize(text)
        # we use this to know which embeddings to use as output of the LLM
        self.clip.tokenized_text_new = text

        # embed text to extract SOS, CLS and EOS tokens (that are not learnabled)
        with torch.no_grad():
            embedding = self.clip.token_embedding(text)
        self.clip.register_buffer("token_prefix_new", embedding[:, :1, :])  # SOS
        self.clip.register_buffer(
            "token_suffix_new", embedding[:, 1 + num_prompts :, :]
        )  # CLS, EOS

        def encode_text(self, split=None):
            assert split in ["base", "new"]

            if split == "base":
                tokenized_text = self.tokenized_text_base
                prompts = self.prompt_embeddings.repeat(len(tokenized_text), 1, 1).to(
                    self.token_prefix_base.device
                )
                x = torch.cat([self.token_prefix_base, prompts, self.token_suffix_base], dim=1)
            elif split == "new":
                tokenized_text = self.tokenized_text_new
                prompts = self.prompt_embeddings.expand(len(tokenized_text), -1, -1).to(
                    self.token_prefix_new.device
                )
                x = torch.cat([self.token_prefix_new, prompts, self.token_suffix_new], dim=1)

            x = x + self.positional_embedding.type(self.dtype)
            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.transformer(x)
            x = x.permute(1, 0, 2)  # LND -> NLD
            x = self.ln_final(x).type(self.dtype)

            # x.shape = [batch_size, n_ctx, transformer.width]
            # take features from the eot embedding (eot_token is the highest
            # number in each sequence)
            x = x[torch.arange(x.shape[0]), tokenized_text.argmax(dim=-1)] @ self.text_projection

            return x

        self.clip.encode_text = types.MethodType(encode_text, self.clip)

    def set_learnable_params(self):
        # turn off all parameters
        for p in self.parameters():
            p.requires_grad = False

        # learnable prompts for the visual model
        if self.cfg.visual_prompts.enabled:
            self.clip.visual.prompt_embeddings.requires_grad = True

        # learnable prompts for the text model
        if self.cfg.text_prompts.enabled:
            self.clip.prompt_embeddings.requires_grad = True

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

    def forward_image(self, x: torch.Tensor):
        with torch.set_grad_enabled(self.cfg.visual_prompts.enabled):
            image_feats = self.clip.visual(x)
        image_feats = image_feats / image_feats.norm(dim=1, keepdim=True)
        return image_feats

    def forward_text(self, tokenized_text, split=None):
        if self.cfg.text_prompts.enabled:
            if self.base_new:
                text_feats = self.clip.encode_text(split)
            else:
                text_feats = self.clip.encode_text()
            text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
        else:
            n_classes, n_prompts = tokenized_text.shape[:2]
            tokenized_text = einops.rearrange(tokenized_text, "c p d -> (c p) d")
            with torch.no_grad():
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
        split: str = "base",
    ):
        image_feats = self.forward_image(x)
        text_feats = self.forward_text(tokenized_text, split)

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

        # normal forward/ce loss
        if self.cfg.text_prompts.enabled:
            tokenized_text = None
        else:
            tokenized_text = self.tokenized_text_base if self.base_new else self.tokenized_text
        logits = self(X, tokenized_text, split="base" if self.base_new else None)
        ce_loss = self.loss_fn(logits, y)

        # auxiliary z-loss
        log_z = torch.logsumexp(logits, dim=-1)
        z_loss = (self.cfg.weights.z_loss * log_z**2).mean()

        if self.synthetic_data:
            logits_synth = self(X_synth, tokenized_text, split="base" if self.base_new else None)
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
        if self.cfg.text_prompts.enabled:
            tokenized_text = None
            if self.base_new:
                split = "base" if dataloader_idx == 0 else "new"
            else:
                split = None
        else:
            split = None
            if self.base_new:
                tokenized_text = (
                    self.tokenized_text_base if dataloader_idx == 0 else self.tokenized_text_new
                )
            else:
                tokenized_text = self.tokenized_text

        logits = self(X, tokenized_text, split=split)

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
