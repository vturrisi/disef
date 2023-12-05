import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

import torch
from lightning.pytorch.utilities.combined_loader import CombinedLoader
from PIL import Image
from torchvision.datasets.vision import VisionDataset

__all__ = ["ClassificationDataset", "SyntheticDataset"]


def default_loader(path: str) -> Any:
    """Loads an image from a path.

    Args:
        path (str): str to the image.

    Returns:
        PIL.Image: The image.
    """
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


class ClassificationDataset(VisionDataset):
    """Dataset for image classification.

    If only the root directory is provided, the dataset works as the `ImageFolder` dataset from
    torchvision. It is otherwise possible to provide a list of images and/or labels. To modify
    the class names, it is possible to provide a list of class names. If the class names are not
    provided, they are inferred from the folder names.

    Args:
        root (str): Root directory of dataset where `images` are found.
        images (list[str], optional): List of images. Defaults to None.
        labels (list[int] | list[list[int]], optional): List of labels (supports multi-labels).
            Defaults to None.
        class_names (list[str], optional): List of class names. Defaults to None.
        transform (Callable | list[Callable], optional): A function/transform that takes in a
            PIL image and returns a transformed version. If a list of transforms is provided, they
            are applied depending on the target label. Defaults to None.
        target_transform (Callable, optional): A function/transform that takes in the target and
            transforms it.
        split (str): which datasplit to use, either train, val or test.
        k_shot (int): number of shots for few-shot.
        seed (int): seed for sampling the k-shots.
        subsample (str): either to use "all" classes, "base" classes or "new" classes.

     Attributes:
        class_names (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index, domain_index) tuples.
        images (list): List of paths to images.
        targets (list): The class_index value for each image in the dataset.
    """

    def __init__(
        self,
        root: str,
        images: Optional[list[str]] = None,
        labels: Optional[Union[list[int], list[list[int]]]] = None,
        class_names: Optional[list[str]] = None,
        transform: Optional[Union[Callable, list[Callable]]] = None,
        target_transform: Optional[Callable] = None,
        split: str = "train",
        k_shot: Optional[int] = None,
        seed: Optional[int] = 5,
        subsample: Optional[str] = "all",
    ) -> None:
        if not images:
            images = sorted([str(path) for path in Path(root).glob("*/*")])

        if not class_names:
            class_names = {Path(f).parent.name for f in images}

        if not labels:
            folder_names = {Path(f).parent.name for f in images}
            folder_names = sorted(folder_names)
            folder_names_to_idx = {c: i for i, c in enumerate(folder_names)}
            labels = [folder_names_to_idx[Path(f).parent.name] for f in images]

        self.samples = list(zip(images, labels))
        self.images = images
        self.targets = labels

        if split == "train" and k_shot:
            self.samples = self.prepare_k_shot(k_shot, seed)
            self.images = [s[0] for s in self.samples]
            self.targets = [s[1] for s in self.samples]

        if subsample in ["base", "new"]:
            # class names are already subsampled and we have class names aligned
            # with the class indexes. So, we just need to split accordingly in a hacky way :(
            split = len(class_names)
            if subsample == "base":
                self.samples = [sample for sample in self.samples if sample[1] < split]
                self.targets = [s[1] for s in self.samples]
            elif subsample == "new":
                split = max(self.targets) + 1 - split
                self.samples = [
                    [sample[0], sample[1] - split] for sample in self.samples if sample[1] >= split
                ]

            self.images = [s[0] for s in self.samples]
            self.targets = [s[1] for s in self.samples]

        self.is_multi_label = all(isinstance(t, list) for t in labels)

        self.class_names = class_names
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.split = split

        self.loader = default_loader

    def prepare_k_shot(self, k_shot: int, seed: int):
        old_state = random.getstate()

        random.seed(seed)
        indexes_perclass = defaultdict(list)

        for i, (_, y) in enumerate(self.samples):
            indexes_perclass[y].append(i)

        selected_indexes = []
        for y, indexes in indexes_perclass.items():
            selected_indexes.extend(random.sample(indexes, k=k_shot))
        data = [self.samples[index] for index in selected_indexes]

        random.setstate(old_state)

        return data

    def __getitem__(self, index: int):
        path, target_idx = self.samples[index]

        image_pil = self.loader(path)
        if self.transform is not None:
            image_tensor = self.transform(image_pil)

        return image_tensor, target_idx

    def __len__(self):
        return len(self.samples)

    def __repr__(self) -> str:
        head = "Dataset " + self.__class__.__name__
        body = [f"Number of datapoints: {self.__len__()}"]
        if self.root is not None:
            body.append(f"Root location: {self.root}")
        if self.class_names is not None:
            if len(self.class_names) > 10:
                body += [f"Classes: {', '.join(self.class_names[:10])}..."]
            else:
                body += [f"Classes: {', '.join(self.class_names)}"]
        if hasattr(self, "transform") and self.transform is not None:
            body += [repr(self.transform)]
        lines = [head] + ["    " + line for line in body]
        return "\n".join(lines)


class SyntheticDataset(VisionDataset):
    """Synthetic dataset class for loading additional data generated by our SAP.

    Args:
        data_dir (str): data folder.
        class_to_idx (dict): a mapping of class name to class index.
        transform (Callable): data augmentations for the synthetic data.
        maximum_synthetic_samples (int): maximum number of synthetic samples to use.
            Leave it as -1 to use all samples in the folder.
        subsample (str): either to use "all" classes, "base" classes or "new" classes.
        class_names (list): list of class names, used for subsampling.
    """

    def __init__(
        self,
        data_dir: str,
        class_to_idx: Optional[Dict[str, int]] = None,
        transform: Optional[Union[Callable, list[Callable]]] = None,
        maximum_synthetic_samples: Optional[int] = -1,
        subsample: Optional[str] = "all",
        class_names: Optional[list[str]] = None,
    ):
        if class_to_idx is None:
            class_folders = {class_folder for class_folder in os.listdir(data_dir)}
            class_to_idx = {c: i for i, c in enumerate(sorted(class_folders))}

        self.transform = transform
        self.loader = default_loader

        self.samples = []
        self.images = []
        self.targets = []

        extensions = [".jpg", ".jpeg", ".png"]

        data_dir = Path(data_dir)
        for class_folder in os.listdir(data_dir):
            for i, image in enumerate(
                [
                    image
                    for image in os.listdir(data_dir / class_folder)
                    if any(image.endswith(ext) for ext in extensions)
                ]
            ):
                if maximum_synthetic_samples != -1 and i >= maximum_synthetic_samples:
                    break

                self.samples.append(
                    (str(data_dir / class_folder / image), class_to_idx[class_folder])
                )
                self.images.append(str(data_dir / class_folder / image))
                self.targets.append(class_to_idx[class_folder])

        if subsample in ["base", "new"]:
            # class names are already subsampled and we have class names aligned
            # with the class indexes. So, we just need to split accordingly in a hacky way :(
            split = len(class_names)
            if subsample == "base":
                self.samples = [sample for sample in self.samples if sample[1] < split]
                self.targets = [s[1] for s in self.samples]
            elif subsample == "new":
                split = max(self.targets) + 1 - split
                self.samples = [
                    [sample[0], sample[1] - split] for sample in self.samples if sample[1] >= split
                ]

            self.images = [s[0] for s in self.samples]
            self.targets = [s[1] for s in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        path, targets_idx = self.samples[index]

        image_pil = self.loader(path)
        if self.transform is not None:
            image_tensor = self.transform(image_pil)

        return image_tensor, targets_idx


class ConcatRepeatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i % len(d)] for d in self.datasets)

    def __len__(self):
        return max(len(d) for d in self.datasets)


class ConcatMinDataset(CombinedLoader):
    def __next__(self):
        batches = super().__next__()
        return tuple(b for b in batches)
