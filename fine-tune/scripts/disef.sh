CUDA_VISIBLE_DEVICES=0
dataset="eurosat"
# optimizer
weight_decay=0.001
epochs=50

# scheduler
sched="warmup_cosine"
warmup_epochs=0
warmup_start_lr=1e-5
min_lr=1e-7

# augmentations
augment=True        # True works better for some datasets
mixup=0.1           # 0.1/0.8 also a possibility
cutmix=0.1          # 0.1/1.0 also a possibility
label_smoothing=0.1 # 0.1

# lora
lora_r=16
lora_alpha=32
lora_dropout=0.1
lora_start_block=0
lora_enabled=$((lora_r > 0))

lora_llm_r=16
lora_llm_alpha=32
lora_llm_dropout=0.1
lora_llm_start_block=0
lora_llm_enabled=$((lora_llm_r > 0))

# use synthetic data
synthetic=False
real_ce=0.8
synthetic_ce=0.2

lora_rs=(16 64 64)
lora_alphas=(32 32 64)
real_ce=0.8
synthetic_ce=0.2

for base_new in False; do
    for lr in 0.00003052 0.00006104 0.0001221 0.0002441 0.0004883 0.0009766; do
        for batch_size in 64; do
            for seed in 1 5 10; do
                synthetic_data_dir=/path/to/synthetic/images/$dataset/seed_$seed

                CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python3 main.py \
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
                    data.base_new=$base_new \
                    check_val_every_n_epoch=10 \
                    data.synthetic_data_dir=$synthetic_data_dir \
                    weights.real_ce=$real_ce \
                    weights.synthetic_ce=$synthetic_ce \
                    data.concat_mode="min" \
                    wandb.project="few-shot"
            done
        done
    done
done
