#!/bin/bash
#SBATCH --array=0-14399
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

# lora
lora_r=64
lora_alpha=32
lora_dropout=0.1
lora_start_block=0
lora_enabled=$((lora_r > 0))

lora_llm_r=16
lora_llm_alpha=32
lora_llm_dropout=0.1
lora_llm_start_block=0
lora_llm_enabled=$((lora_llm_r > 0))

# use synthetic data, set to False for real data only (DISEF w/o synthetic)
synthetic=True
synth_samples=64
base_new=False

mixups=(0.1 0.8)
cutmixs=(0.1 1.0)

real_ces=(0.5 0.6 0.7 0.8)
synthetic_ces=(0.5 0.4 0.3 0.2)

#####################################

datasets=("dtd" "eurosat" "oxford_pets" "caltech101" "stanford_cars" "flowers102" "food101" "fgvc_aircraft" "imagenet" "sun397")
lrs=(0.00003052 0.00006104 0.0001221 0.0002441 0.0004883 0.0009766)
batch_sizes=(8 16 32 64 128)
augments=(True False)
mixup_cutmix_set=(0 1)
losses_set=(0 1 2 3)
seeds=(1 5 10)

dataset_idx=$(((($SLURM_ARRAY_TASK_ID) / (${#batch_sizes[@]} * ${#seeds[@]} * ${#lrs[@]} * ${#augments[@]} * ${#mixup_cutmix_set[@]} * ${#losses_set[@]})) % ${#datasets[@]}))
lr_idx=$(((($SLURM_ARRAY_TASK_ID) / (${#batch_sizes[@]} * ${#seeds[@]} * ${#augments[@]} * ${#mixup_cutmix_set[@]} * ${#losses_set[@]})) % ${#lrs[@]}))
batch_size_idx=$(((($SLURM_ARRAY_TASK_ID) / (${#seeds[@]} * ${#augments[@]} * ${#mixup_cutmix_set[@]} * ${#losses_set[@]})) % ${#batch_sizes[@]}))
augment_idx=$(((($SLURM_ARRAY_TASK_ID) / (${#seeds[@]} * ${#mixup_cutmix_set[@]} * ${#losses_set[@]})) % ${#augments[@]}))
mixup_cutmix_idx=$(((($SLURM_ARRAY_TASK_ID) / (${#seeds[@]} * ${#losses_set[@]})) % ${#mixup_cutmix_set[@]}))
losses_idx=$((($SLURM_ARRAY_TASK_ID) % ${#losses_set[@]}))
seed_idx=$((($SLURM_ARRAY_TASK_ID) % ${#seeds[@]}))

dataset=${datasets[$dataset_idx]}
lr=${lrs[$lr_idx]}
batch_size=${batch_sizes[$batch_size_idx]}
augment=${augments[$augment_idx]}
mixup_cutmix=${mixup_cutmix_set[$mixup_cutmix_idx]}
mixup=${mixups[$mixup_cutmix]}
cutmix=${cutmixs[$mixup_cutmix]}
losses=${losses_set[$losses_idx]}
real_ce=${real_ces[$losses]}
synthetic_ce=${synthetic_ces[$losses]}
seed=${seeds[$seed_idx]}

synthetic_data_dir=/path/to/synthetic/images/$dataset/seed_$seed

echo "dataset: $dataset, diffusion_model: $diffusion_model, steps: $steps, seed: $seed, lr: $lr, batch_size: $batch_size, augment: $augment, mixup: $mixup, cutmix: $cutmix, real_ce: $real_ce, synthetic_ce: $synthetic_ce"

python3 main.py \
    --cfg configs/generic.yaml \
    optim.lr=$lr \
    optim.weight_decay=$weight_decay \
    max_epochs=$epochs \
    data.name=$dataset \
    data.batch_size=$batch_size \
    data.val_batch_size=64 \
    name="$dataset-base_new=$base_new-lora-16-shot-bs=$batch_size-lr=$lr-wd=$weight_decay-ep=$epochs-sched=$sched-warmup_epochs=$warmup_epochs-mixup=$mixup-cutmix=$cutmix-label_smoothing=$label_smoothing-synthetic=$synthetic-real_ce=$real_ce-synthetic_ce=$synthetic_ce-augment=$augment-img2img-realistic-step15-highres-lora_r=$lora_r-lora_alpha=$lora_alpha" \
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
    lora.r=$lora_r \
    lora.alpha=$lora_alpha \
    lora.dropout=$lora_dropout \
    lora.start_block=$lora_start_block \
    lora_llm.enabled=$lora_llm_enabled \
    lora_llm.r=$lora_llm_r \
    lora_llm.alpha=$lora_llm_alpha \
    lora_llm.dropout=$lora_llm_dropout \
    lora_llm.start_block=$lora_llm_start_block \
    seed=$seed \
    data.synthetic=$synthetic \
    data.synthetic_data_dir=$synthetic_data_dir \
    data.maximum_synthetic_samples=$synth_samples \
    weights.real_ce=$real_ce \
    weights.synthetic_ce=$synthetic_ce \
    data.base_new=$base_new \
    check_val_every_n_epoch=10 \
    wandb.project="few-shot"
