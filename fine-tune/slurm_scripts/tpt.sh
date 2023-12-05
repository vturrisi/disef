#!/bin/bash
#SBATCH --array=0-7199
#SBATCH --time=0-18:00:00
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --out=slurm_logs/%j.out

# optimizer
weight_decay=0.001
epochs=50

# scheduler
sched="warmup_cosine"
warmup_epochs=0
warmup_start_lr=1e-5
min_lr=1e-7

# augmentations
label_smoothing=0.1

# turn to True to use synthetic data and set the other parameters
synthetic=False
synth_samples=0
synthetic_data_dir=""
real_weight=1.0
synth_weight=0.0

base_new=False

mixups=(0.1 0.8)
cutmixs=(0.1 1.0)

mixup_cutmix=0
mixup=${mixups[$mixup_cutmix]}
cutmix=${cutmixs[$mixup_cutmix]}

# disable visual prompts
vp=0
vp_enabled=$((vp > 0))

# disable lora
lora_enabled=0
lora_llm_enabled=0

#####################################

datasets=("dtd" "eurosat" "oxford_pets" "caltech101" "food101" "flowers102" "stanford_cars" "fgvc_aircraft" "imagenet" "sun397")
lrs=(0.00003052 0.00006104 0.0001221 0.0002441 0.0004883 0.0009766)
augments=(True False)
batch_sizes=(8 16 32 64 128)
text_prompts=(2 4 8 16)
seeds=(1 5 10)

dataset_idx=$(((($SLURM_ARRAY_TASK_ID) / (${#batch_sizes[@]} * ${#seeds[@]} * ${#lrs[@]} * ${#augments[@]} * ${#text_prompts[@]})) % ${#datasets[@]}))
lr_idx=$(((($SLURM_ARRAY_TASK_ID) / (${#batch_sizes[@]} * ${#seeds[@]} * ${#augments[@]} * ${#text_prompts[@]})) % ${#lrs[@]}))
augment_idx=$(((($SLURM_ARRAY_TASK_ID) / (${#batch_sizes[@]} * ${#seeds[@]} * ${#text_prompts[@]})) % ${#augments[@]}))
batch_size_idx=$(((($SLURM_ARRAY_TASK_ID) / (${#seeds[@]} * ${#text_prompts[@]})) % ${#batch_sizes[@]}))
text_prompt_idx=$(((($SLURM_ARRAY_TASK_ID) / ${#seeds[@]}) % ${#text_prompts[@]}))
seed_idx=$((($SLURM_ARRAY_TASK_ID) % ${#seeds[@]}))

dataset=${datasets[$dataset_idx]}
lr=${lrs[$lr_idx]}
augment=${augments[$augment_idx]}
batch_size=${batch_sizes[$batch_size_idx]}
tp=${text_prompts[$text_prompt_idx]}
seed=${seeds[$seed_idx]}

tp_enabled=$((tp > 0))

echo "dataset: $dataset, augment: $augment, mixup: $mixup, cutmix: $cutmix, lr: $lr, batch_size: $batch_size, seed: $seed, text_prompts: $tp"

python3 main.py \
    --cfg configs/generic.yaml \
    optim.lr=$lr \
    optim.weight_decay=$weight_decay \
    max_epochs=$epochs \
    data.name=$dataset \
    data.batch_size=$batch_size \
    data.val_batch_size=64 \
    name="$dataset-base_new=$base_new-tpt-16-shot-bs=$batch_size-lr=$lr-wd=$weight_decay-ep=$epochs-sched=$sched-warmup_epochs=$warmup_epochs-mixup=$mixup-cutmix=$cutmix-label_smoothing=$label_smoothing-tp=$tp_enabled($tp)-synthetic=$synthetic" \
    mixing.enabled=True \
    mixing.mixup=$mixup \
    mixing.cutmix=$cutmix \
    mixing.label_smoothing=$label_smoothing \
    data.augment=$augment \
    sched.name=$sched \
    sched.warmup_epochs=$warmup_epochs \
    sched.warmup_start_lr=$warmup_start_lr \
    sched.min_lr=$min_lr \
    sched.interval="step" \
    lora.enabled=$lora_enabled \
    lora_llm.enabled=$lora_llm_enabled \
    seed=$seed \
    data.synthetic=$synthetic \
    data.synthetic_data_dir=$synthetic_data_dir \
    data.maximum_synthetic_samples=$synth_samples \
    weights.real_ce=$real_weight \
    weights.synthetic_ce=$synth_weight \
    data.base_new=$base_new \
    visual_prompts.enabled=$vp_enabled \
    visual_prompts.number=$vp \
    text_prompts.enabled=$tp_enabled \
    text_prompts.number=$tp \
    check_val_every_n_epoch=10 \
    wandb.project="few-shot"
