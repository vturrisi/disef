from pathlib import Path
from typing import Optional

from src.data._base import BaseDataModule
from src.data.components.datasets import ClassificationDataset, SyntheticDataset


class FGVCAircraft(BaseDataModule):
    """LightningDataModule for FGVCAircraft dataset.

    Statistics:
        - Around 10,000 images.
        - 100 classes.
        - URL: https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/.

    Reference:
        - Maji et al.  Fine-Grained Visual Classification of Aircraft. Preprint 2013.

    Args:
        data_dir (str): Path to the data directory. Defaults to "data/".
        artifact_dir (str): Path to the artifacts directory. Defaults to "artifacts/".
    """

    name: str = "FGVCAircraft"
    task: str = "classification"

    classes: list[str]

    data_url: str = (
        "https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz"
    )

    def __init__(self, *args, data_dir: str = "data/", **kwargs):
        super().__init__(*args, data_dir=data_dir, **kwargs)

    @property
    def classes(self):
        if not self._classes:
            dataset_path = Path(self.hparams.data_dir, self.name)
            metadata_fp = dataset_path / "variants.txt"
            class_names = [line.strip() for line in open(metadata_fp).readlines()]
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

        # read folder names
        dataset_path = Path(self.hparams.data_dir, self.name)
        metadata_fp = dataset_path / "variants.txt"
        class_names = [line.strip() for line in open(metadata_fp).readlines()]
        classes_to_idx = {c: i for i, c in enumerate(class_names)}

        # some settings
        k_shot = self.hparams.get("k_shot", None)
        seed = self.hparams.get("seed", 5)
        subsample = self.hparams.get("subsample", "all")

        data = {}
        for split in ["train", "val", "test"]:
            split_fp = dataset_path / f"images_variant_{split}.txt"
            with open(split_fp) as f:
                lines = [line.strip() for line in f.readlines()]
                filenames = [line.split(" ")[0] for line in lines]
                labels = [" ".join(line.split(" ")[1:]) for line in lines]
            image_paths = [str(dataset_path / "images" / f"{x}.jpg") for x in filenames]
            labels = [classes_to_idx[c] for c in labels]

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
    _ = FGVCAircraft()
