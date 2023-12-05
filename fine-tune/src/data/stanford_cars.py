from pathlib import Path
from typing import Optional

import pandas as pd
from src.data._base import BaseDataModule
from src.data.components.datasets import ClassificationDataset, SyntheticDataset


class StanfordCars(BaseDataModule):
    """LightningDataModule for StanfordCars dataset.

    Statistics:
        - 16,185 images.
        - 196 classes.
        - URL: https://ai.stanford.edu/~jkrause/cars/car_dataset.html.

    Reference:
        - Krause et al. Cars: Categorization and Detection. CVPR 2013.

    Args:
        data_dir (str): Path to the data directory. Defaults to "data/".
        artifact_dir (str): Path to the artifacts directory. Defaults to "artifacts/".
    """

    name: str = "StanfordCars"
    task: str = "classification"

    classes: list[str]

    data_url: dict[str, str] = {
        "train": "http://ai.stanford.edu/~jkrause/car196/cars_train.tgz",
        "test": "http://ai.stanford.edu/~jkrause/car196/cars_test.tgz",
    }

    def __init__(self, *args, data_dir: str = "data/", artifact_dir: str = "artifacts/", **kwargs):
        super().__init__(*args, data_dir=data_dir, **kwargs)

    @property
    def classes(self):
        if not self._classes:
            metadata_fp = Path(self.hparams.artifact_dir, "stanford_cars", "metadata.csv")
            metadata_df = pd.read_csv(metadata_fp)
            class_names = metadata_df["class_name"].tolist()
            self._classes = class_names
            self._classes = self.subsample_classes(self._classes)
        return self._classes

    def setup(self, stage: Optional[str] = None):
        """Load data.

        Set variables: `self.data_train` , `self.data_val` and `self.data_test`.
        """
        if self.data_train and self.data_val and self.data_test:
            return

        # setup paths
        dataset_path = Path(self.hparams.data_dir, self.name)
        labels_fp = Path(self.hparams.artifact_dir, "stanford_cars", "labels.csv")
        split_fp = Path(self.hparams.artifact_dir, "stanford_cars", "split_coop.csv")

        # read labels df names
        labels_df = pd.read_csv(labels_fp)

        # some settings
        k_shot = self.hparams.get("k_shot", None)
        seed = self.hparams.get("seed", 5)
        subsample = self.hparams.get("subsample", "all")

        # load data info
        split_df = pd.read_csv(split_fp)

        data = {}
        for split in ["train", "val", "test"]:
            image_paths = split_df[split_df["split"] == split]["filename"]
            merge_df = pd.merge(image_paths, labels_df, on="filename")
            image_paths = merge_df["filename"]
            image_paths = image_paths.apply(lambda x: str(dataset_path / x)).tolist()
            labels = merge_df["class_idx"].tolist()

            if split == "train" and self.hparams.get("synthetic", False):
                real_dataset = ClassificationDataset(
                    str(dataset_path),
                    images=image_paths,
                    labels=labels,
                    class_names=self.classes,
                    transform=self.get_transforms(split, self.hparams.get("augment", True)),
                    split=split,
                    k_shot=k_shot,
                    seed=seed,
                    subsample=subsample,
                )
                metadata_fp = Path(self.hparams.artifact_dir, "stanford_cars", "metadata.csv")
                metadata_df = pd.read_csv(metadata_fp)
                classes_to_idx = {
                    class_name: i for i, class_name in enumerate(metadata_df["class_name"].tolist())
                }
                synthetic_dataset = SyntheticDataset(
                    self.hparams.get("synthetic_data_dir", None),
                    classes_to_idx,
                    transform=self.get_transforms(split, self.hparams.get("augment", True)),
                    maximum_synthetic_samples=self.hparams.get("maximum_synthetic_samples", -1),
                    class_names=self.classes,
                    subsample=subsample,
                )
                data[split] = [real_dataset, synthetic_dataset]
            else:
                data[split] = ClassificationDataset(
                    str(dataset_path),
                    images=image_paths,
                    labels=labels,
                    class_names=self.classes,
                    transform=self.get_transforms(split, self.hparams.get("augment", True)),
                    split=split,
                    k_shot=k_shot,
                    seed=seed,
                    subsample=subsample,
                )

        self.data_train = data["train"]
        self.data_val = self.data_test = data["test"]


if __name__ == "__main__":
    _ = StanfordCars()
