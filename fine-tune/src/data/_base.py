import math
from abc import ABC, abstractmethod
from typing import Optional

from lightning import LightningDataModule
from PIL import Image
from src.data.components.datasets import ConcatRepeatDataset, ConcatMinDataset
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandAugment,
    RandomResizedCrop,
    Resize,
    ToTensor,
)
from torch.utils.data import ConcatDataset


class BaseDataModule(ABC, LightningDataModule):
    """LightningDataModule with base functionalities.

    Args:
        data_dir (str): Path to data directory.
        train_val_split (tuple[float, float]): Train/val split.

    Extra hparams:
        batch_size (int): Batch size. Defaults to 64.
        num_workers (int): Number of workers. Defaults to 0.
        pin_memory (bool): Pin memory. Defaults to False.
        image_size (int): Image size. Default to 224.
        train_cycle_mode (str): Train cycle mode. Default to "max_size_cycle".

    Attributes:
        name (str): Name of the dataset.
        classes (list[str]): List of class names.
        task (str): Task of the dataset.
        data_train (Dataset): Training dataset.
        data_val (Dataset): Validation dataset.
        data_test (Dataset): Test dataset.
        num_classes (int): Number of classes.
    """

    name: str
    task: str

    classes: list[str]

    def __init__(
        self,
        *args,
        data_dir: str = "data/",
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["_metadata_"])

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        self._classes = None

    @property
    def num_classes(self):
        return len(self.classes)

    def subsample_classes(self, classes):
        subsample = self.hparams.get("subsample", "all")
        assert subsample in ["all", "base", "new"]
        split = math.ceil(len(classes) / 2)
        if subsample == "base":
            return classes[:split]
        elif subsample == "new":
            return classes[split:]
        else:
            return classes

    def subsample_data(self, images, labels, classes):
        subsampled_images = []
        subsampled_classes = []
        for image, c in zip(images, labels):
            if c in classes:
                subsampled_images.append(image)
                subsampled_classes.append(c)
        return subsampled_images, subsampled_classes

    def get_transforms(self, split: str, augment_data: bool):
        if self.hparams.get("classifier_tuning", False):
            print("Using Classifier Tuning augmentations")
            if split == "train" and augment_data:
                transforms = Compose(
                    [
                        RandAugment(),
                        RandomResizedCrop(224, interpolation=Image.BICUBIC),
                        ToTensor(),
                        Normalize(
                            (0.48145466, 0.4578275, 0.40821073),
                            (0.26862954, 0.26130258, 0.27577711),
                        ),
                    ]
                )
            else:
                transforms = Compose(
                    [
                        Resize(256, interpolation=Image.BICUBIC),
                        CenterCrop(224),
                        ToTensor(),
                        Normalize(
                            (0.48145466, 0.4578275, 0.40821073),
                            (0.26862954, 0.26130258, 0.27577711),
                        ),
                    ]
                )
        else:
            print("Using default augmentations")
            if split == "train" and augment_data:
                transforms = Compose(
                    [
                        RandAugment(
                            self.hparams.get("randaugment_n", 2),
                            self.hparams.get("randaugment_m", 9),
                        ),
                        RandomResizedCrop(224, interpolation=Image.BICUBIC),
                        ToTensor(),
                        Normalize(
                            (0.48145466, 0.4578275, 0.40821073),
                            (0.26862954, 0.26130258, 0.27577711),
                        ),
                    ]
                )
            elif split == "train":
                transforms = Compose(
                    [
                        RandomResizedCrop(224, interpolation=Image.BICUBIC),
                        ToTensor(),
                        Normalize(
                            (0.48145466, 0.4578275, 0.40821073),
                            (0.26862954, 0.26130258, 0.27577711),
                        ),
                    ]
                )
            else:
                transforms = Compose(
                    [
                        Resize(256, interpolation=Image.BICUBIC),
                        CenterCrop(224),
                        ToTensor(),
                        Normalize(
                            (0.48145466, 0.4578275, 0.40821073),
                            (0.26862954, 0.26130258, 0.27577711),
                        ),
                    ]
                )
        return transforms

    @abstractmethod
    def setup(self, stage: Optional[str] = None):
        """Load data.

        Set variables: `self.data_train`, `self.data_val`, `self.data_test`.
        """
        raise NotImplementedError

    def dataloader_kwargs(self, split: str):
        """Get default kwargs for dataloader."""
        return {
            "batch_size": self.hparams.get("batch_size", 64)
            if split == "train"
            else self.hparams.get("val_batch_size", 64),
            "num_workers": self.hparams.get("num_workers", 0),
            "pin_memory": self.hparams.get("pin_memory", True),
            "drop_last": split == "train",
        }

    def train_dataloader(self):
        if isinstance(self.data_train, list):
            concat_mode = self.hparams.get("concat_mode", "max")
            if concat_mode == "max":
                return DataLoader(
                    dataset=ConcatRepeatDataset(self.data_train[0], self.data_train[1]),
                    **self.dataloader_kwargs("train"),
                    shuffle=True,
                )
            elif concat_mode == "min":
                return ConcatMinDataset(
                    [
                        DataLoader(
                            dataset=dataset,
                            **self.dataloader_kwargs("train"),
                            shuffle=True,
                        )
                        for dataset in self.data_train
                    ],
                    mode="min_size",
                )
            elif concat_mode == "sequential":
                return DataLoader(
                    dataset=ConcatDataset([self.data_train[0], self.data_train[1]]),
                    **self.dataloader_kwargs("train"),
                    shuffle=True,
                )
            else:
                raise ValueError(
                    f"concat_mode={concat_mode} not supported. Choose from [min, max, sequential]"
                )

        return DataLoader(
            dataset=self.data_train,
            **self.dataloader_kwargs("train"),
            shuffle=True,
        )

    def val_dataloader(self):
        if isinstance(self.data_val, list):
            return [
                DataLoader(dataset=data_val, **self.dataloader_kwargs("val"), shuffle=False)
                for data_val in self.data_val
            ]

        return DataLoader(dataset=self.data_val, **self.dataloader_kwargs("val"), shuffle=False)

    def test_dataloader(self):
        if isinstance(self.data_test, list):
            return [
                DataLoader(dataset=data_test, **self.dataloader_kwargs("test"), shuffle=False)
                for data_test in self.data_test
            ]

        return DataLoader(dataset=self.data_test, **self.dataloader_kwargs("test"), shuffle=False)
