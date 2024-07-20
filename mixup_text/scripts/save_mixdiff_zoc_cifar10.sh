#!/bin/bash

for method in \
    MixDiffZOC
do 
    for dataset in \
        CIFAR10OODDataset 
    do
        python -m mixup.save_mixup_scores \
            --n 10 \
            --m 10 \
            --r 7 \
            --gamma 1.0 \
            --seed 0 \
            --wandb_name logging \
            --device 4 \
            --max_samples 200 \
            --truncate_by_max_samples false \
            --shuffle false \
            --score_calculator.class_path mixup.ood_score_calculators.$method \
            --score_calculator.init_args.batch_size 150 \
            --score_calculator.init_args.zoc_checkpoint_path trained_models/COCO/ViT-B32/model.pt \
            --score_calculator.init_args.utilize_mixup true \
            --score_calculator.init_args.add_base_scores true \
            --score_calculator.init_args.follow_zoc true \
            --score_calculator.init_args.half_precision true \
            --score_calculator.init_args.avg_logits exp \
            --datamodule.class_path mixup.ood_datamodules.$dataset 
    done
done