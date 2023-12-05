from pathlib import Path
from typing import Optional

import pandas as pd
from src.data._base import BaseDataModule
from src.data.components.datasets import ClassificationDataset, SyntheticDataset


class Caltech101(BaseDataModule):
    """LightningDataModule for Caltech-101 dataset.

    Statistics:
        - Around 9,000 images.
        - 100 classes.
        - URL: https://data.caltech.edu/records/mzrjq-6wc02.

    Reference:
        - Li et al. One-shot learning of object categories. TPAMI 2006.

    Args:
        data_dir (str): Path to the data directory. Defaults to "data/".
        artifact_dir (str): Path to the artifacts directory. Defaults to "artifacts/".
    """

    name: str = "Caltech101"
    task: str = "classification"

    classes: list[str]

    data_url: str = "https://data.caltech.edu/records/mzrjq-6wc02/files/caltech-101.zip"

    def __init__(self, *args, data_dir: str = "data/", artifact_dir: str = "artifacts/", **kwargs):
        super().__init__(*args, data_dir=data_dir, **kwargs)

    @property
    def classes(self):
        if not self._classes:
            metadata_fp = Path(self.hparams.artifact_dir, "caltech101", "metadata.csv")
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
        metadata_fp = Path(self.hparams.artifact_dir, "caltech101", "metadata.csv")
        split_fp = Path(self.hparams.artifact_dir, "caltech101", "split_coop.csv")

        # read folder names
        metadata_df = pd.read_csv(metadata_fp)
        classes_to_idx = {str(c): i for i, c in enumerate(metadata_df["folder_name"].tolist())}

        # some settings
        k_shot = self.hparams.get("k_shot", None)
        seed = self.hparams.get("seed", 5)
        subsample = self.hparams.get("subsample", "all")

        # load data info
        split_df = pd.read_csv(split_fp)

        data = {}
        for split in ["train", "val", "test"]:
            image_paths = split_df[split_df["split"] == split]["filename"]
            image_paths = image_paths.apply(lambda x: str(dataset_path / x)).tolist()
            folder_names = [Path(f).parent.name for f in image_paths]
            labels = [classes_to_idx[c] for c in folder_names]

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
    _ = Caltech101()
