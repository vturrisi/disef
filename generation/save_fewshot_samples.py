from argparse import ArgumentParser
import os

from lightning.pytorch import seed_everything
from omegaconf import OmegaConf
from src.data import DATA
from src.utils.misc import omegaconf_select


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
    dataset = cfg.data.name

    datamodule = DATA[cfg.data.name.lower()](
        k_shot=cfg.data.k_shot,
        smaller_k_shot=omegaconf_select(cfg.data, "smaller_k_shot", None),
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        synthetic=omegaconf_select(cfg.data, "synthetic", False),
        synthetic_data_dir=omegaconf_select(cfg.data, "synthetic_data_dir", None),
        maximum_synthetic_samples=omegaconf_select(cfg.data, "maximum_synthetic_samples", -1),
        augment=omegaconf_select(cfg.data, "augment", True),
        subsample=omegaconf_select(cfg.data, "subsample", "all"),
        ct=omegaconf_select(cfg.data, "ct", False),
        seed=seed,
    )

    datamodule.setup()

    
    os.makedirs("image_paths", exist_ok=True)
    
    # get the filenames for the images in the dataset and save them to a file
    for split, images in zip(
        ["train"], [datamodule.data_train.images],
    ):
        with open(f"image_paths/{dataset}-{split}-16shots-seed={seed}.txt", "w") as f:
            for image in images:
                f.write(f"{image}\n")

    exit()


if __name__ == "__main__":
    main()
