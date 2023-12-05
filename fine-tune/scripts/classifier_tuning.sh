for dataset in "caltech101" "dtd" "eurosat" "oxford_pets" "stanford_cars" "fgvc_aircraft" "sun397" "imagenet"; do
    for batch_size in 32 64 128; do
        for lr in 0.05 0.03 0.01; do
            for seed in 1 5 10; do
                for augment in True False; do
                    python3 main_ct.py \
                        --cfg configs/classifier_tuning.yaml \
                        optim.lr=$lr \
                        data.batch_size=$batch_size \
                        data.augment=$augment \
                        data.name=$dataset \
                        name="$dataset-classifier-tuning-16-shot-bs=$batch_size-lr=$lr-augment=$augment" \
                        seed=$seed \
                        check_val_every_n_epoch=10 \
                        wandb.project="few-shot"
                done
            done
        done
    done
done
