#!/bin/bash
#SBATCH --array=0-1199
#SBATCH --time=0-18:00:00
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --out=slurm_logs/%j.out

# optimizer
weight_decay=0.1
epochs=50

# scheduler
sched="warmup_cosine"
warmup_epochs=0
warmup_start_lr=1e-5
min_lr=1e-7

# augmentations
label_smoothing=0.0

base_new=False

mixups=(0.1 0.8 0.0)
cutmixs=(0.1 1.0 0.0)

mixing=False
mixup_cutmix=2

# turn to True to use synthetic data and set the other parameters
synthetic=False
synthetic_data_dir=""
synth_samples=0
real_weight=1.0
synth_weight=0.0

#####################################

datasets=("dtd" "eurosat" "oxford_pets" "caltech101" "stanford_cars" "flowers102" "food101" "fgvc_aircraft" "imagenet" "sun397")
lrs=(0.05 0.01 0.005 0.001)
batch_sizes=(8 16 32 64 128)
augments=(True False)
seeds=(1 5 10)

dataset_idx=$(((($SLURM_ARRAY_TASK_ID) / (${#batch_sizes[@]} * ${#seeds[@]} * ${#lrs[@]} * ${#augments[@]})) % ${#datasets[@]}))
lr_idx=$(((($SLURM_ARRAY_TASK_ID) / (${#batch_sizes[@]} * ${#seeds[@]} * ${#augments[@]})) % ${#lrs[@]}))
batch_size_idx=$(((($SLURM_ARRAY_TASK_ID) / (${#seeds[@]} * ${#augments[@]})) % ${#batch_sizes[@]}))
augment_idx=$(((($SLURM_ARRAY_TASK_ID) / ${#seeds[@]}) % ${#augments[@]}))
seed_idx=$((($SLURM_ARRAY_TASK_ID) % ${#seeds[@]}))

dataset=${datasets[$dataset_idx]}
augment=${augments[$augment_idx]}
mixup=${mixups[$mixup_cutmix]}
cutmix=${cutmixs[$mixup_cutmix]}
lr=${lrs[$lr_idx]}
batch_size=${batch_sizes[$batch_size_idx]}
seed=${seeds[$seed_idx]}

echo "dataset: $dataset, augment: $augment, mixup: $mixup, cutmix: $cutmix, lr: $lr, batch_size: $batch_size, seed: $seed"

python3 main_ct.py \
    --cfg configs/classifier_tuning.yaml \
    optim.lr=$lr \
    data.ct=True \
    data.batch_size=$batch_size \
    data.augment=$augment \
    mixing.enabled=$mixing \
    mixing.mixup=$mixup \
    mixing.cutmix=$cutmix \
    mixing.label_smoothing=$label_smoothing \
    data.name=$dataset \
    name="$dataset-classifier-tuning-16-shot-bs=$batch_size-lr=$lr-weight_decay=$weight_decay-augment=$augment-mixing=$mixing-mixup=$mixup-cutmix=$cutmix-label_smoothing=$label_smoothing-real_weight=$real_weight-synth_weight=$synth_weight-synthetic=$synthetic-synth_samples=$synth_samples" \
    data.synthetic=$synthetic \
    data.synthetic_data_dir=$synthetic_data_dir \
    data.maximum_synthetic_samples=$synth_samples \
    weights.real_ce=$real_weight \
    weights.synthetic_ce=$synth_weight \
    seed=$seed \
    wandb.project="few-shot"
