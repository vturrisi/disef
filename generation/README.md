# Synthetic Augmentation Pipeline (SAP)
This is the source code for generating synthetic data with SAP.

## Installation
For installation, we suggest using conda to keep a clean environment:
```
conda create --name sap python=3.9
conda activate sap
pip3 install -r requirements.txt
```
After that, you will need to install LLaVA 1.5.

## Install LLaVA 1.5

- `git clone https://github.com/haotian-liu/LLaVA.git`
- Move `llava_processor.py` (it enables batch processing) to the LLaVA folder
    - `mv llava_processor.py LLaVA/llava`
- Install LLaVA
    - `cd LLaVA`
    - `pip install -e .`
- Download LLaVA 1.5 7B from [here](https://huggingface.co/liuhaotian/llava-v1.5-7b)

## How to run

### Obtain 16 shot per class for a given dataset and a given seed

- Put `save_fewshot_samples.py` in the finetuning code folder and run:

```bash
dataset=dtd  # eurosat, caltech101, oxford_pets, stanford_cars, fgvc_aircract, food101, flowers102, sun397, imagenet
seed=1  # 5, 10

python save_fewshot_samples.py --cfg configs/generic.yaml data.name=$dataset seed=$seed
```

This will produce a file `image_paths/$dataset-train-16shots-seed=$seed.txt`

- Then move the selected images to a folder with structure like:

```
selected_fewshots/
|- $dataset/
|  |- seed_$seed/
|  |  |- <class_0>
|  |  |  |- {images}.jpg
```

The procedure to have this folder depends on the dataset, most of the datasets already have a class-based structure, refer to the artifacts folder to get the class of each sample otherwise.

### Generation

- Call our `synthetic_augmentation_pipeline` with the following arguments:

```bash
dataset=dtd  # eurosat, caltech101, oxford_pets, stanford_cars, fgvc_aircract, food101, flowers102, sun397, imagenet
seed=1  # 5, 10
steps=5  # dataset-dependant as reported in the paper and supplementary, it refers to the amount of noise added to the latents, a lower value preserves the original image more

python synthetic_augmentation_pipeline.py \ 
    --datasets_root selected_fewshots/ \ 
    --output_dir your/output/dir \ 
    --dataset $dataset \ 
    --seed $seed \ 
    --steps $steps \ 
    --model_name realistic \ 
    --use_llava \ 
    --llava_model_path path/to/download/llava
```

You will find the generated images in `your/output/dir`, and they can be used in the finetuning step.
