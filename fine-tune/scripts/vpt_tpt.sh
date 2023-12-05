CUDA_VISIBLE_DEVICES=0
dataset="eurosat"
# optimizer
batch_size=32
weight_decay=0.001
epochs=50

# scheduler
sched="warmup_cosine"
warmup_epochs=0
warmup_start_lr=1e-5
min_lr=1e-7

# augmentations
augment=False       # True works better for some datasets
mixup=0.1           # 0.0/0.8 also a possibility
cutmix=0.1          # 0.0/1.0 also a possibility
label_smoothing=0.1 # 0.0

# visual learnable prompts
vp=30
vp_enabled=$((vp > 0))

# text learnable prompts
tp=30
tp_enabled=$((tp > 0))

# use synthetic data
synthetic=False

# either use all classes (default) or just the base/new classes
base_new=False

for lr in 0.0005; do
    for batch_size in 32; do
        for seed in 1 5 10; do
            CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python3 main.py \
                --cfg configs/generic.yaml \
                optim.lr=$lr \
                optim.weight_decay=$weight_decay \
                max_epochs=$epochs \
                data.name=$dataset \
                data.batch_size=$batch_size \
                data.val_batch_size=64 \
                name="$dataset-base_new=$base_new-vpt+tpt-16-shot-bs=$batch_size-lr=$lr-wd=$weight_decay-ep=$epochs-sched=$sched-warmup_epochs=$warmup_epochs-mixup=$mixup-cutmix=$cutmix-label_smoothing=$label_smoothing-vp=$vp_enabled($vp)-tp=$tp_enabled($tp)-synthetic=$synthetic" \
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
                visual_prompts.enabled=$vp_enabled \
                visual_prompts.number=$vp \
                text_prompts.enabled=$tp_enabled \
                text_prompts.number=$tp \
                seed=$seed \
                data.synthetic=$synthetic \
                data.base_new=$base_new \
                check_val_every_n_epoch=10 \
                wandb.project="few-shot"
        done
    done
done
