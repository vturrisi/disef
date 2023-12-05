# Overview
This is the source code for fine-tuning a VLM with DISEF.
It contains all the utily functions for dataloading, training, loggings, etc.
It also contains other baselines like VPT, TPT, VPT + TPT and Classifier Tuning.

## Structure
- `artifacts` contain extra information about individual datasets.
- `configs` contain the base yaml configuration files for the experiments.
- `data` should contain all the downloaded datasets.
- `scripts` contain example bash scripts for running different experiments.
- `slurm_scripts` contain the scripts for launching many experiments with slurm. 
- `src` contain the source code for:
    - DISEF, with and without synthetic data, VPT, TPT, VPT + TPT and Classifier Tuning.
    - All dataloaders for the different benchmark datasets.
    - Utility functions.


## Downloading the data

Download the individual datasets by following their respective tutorials and move them to `data`.
Our dataloaders also use the extra dataset info files in `artifacts`, but those are already provided.

- DTD: [here](https://www.robots.ox.ac.uk/~vgg/data/dtd/)
- EuroSAT: [here](https://github.com/phelber/EuroSAT)
- Oxford-IIIT Pet: [here](https://www.robots.ox.ac.uk/~vgg/data/pets/)
- Stanford Cars: [here](https://www.kaggle.com/datasets/jessicali9530/stanford-cars-dataset)
- FGVC Aircraft: [here](https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/)
- Caltech101: [here](https://data.caltech.edu/records/mzrjq-6wc02)
- Food101: [here](https://www.kaggle.com/dansbecker/food-101)
- Flowers102: [here](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)
- SUN397: [here](https://vision.princeton.edu/projects/2010/SUN/)
- ImageNet: [here](https://www.kaggle.com/c/imagenet-object-localization-challenge/data)


## Installation

For installation, we suggest using conda to keep a clean environment:
```
conda create --name disef python=3.9
conda activate disef
pip3 install -r requirements.txt
```

## Running

Example scripts are available in `scripts`. Simply change the parameters there and run as, for example:
```
bash scripts/disef.sh
```

## Running with Slurm

To launch a more experiments, we suggest leveraging slurm.
For that, we provide in `slurm_scripts` all the scripts for running grid search on all datasets and methods.

To launch any experiment, first set the correct slurm options in the header according to your cluster.
After that, set the correct paths for the synthetic data, if using it
(refer to `generation/README` for how to generate the data)
Then, simply launch as:
```
sbatch slurm_scripts/disef.sh
```
