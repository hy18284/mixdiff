#!/bin/bash

for method in \
    MixDiffEnergy
do 
    for dataset in \
        CIFAR10OODDataset 
    do
        python -m mixup.save_mixup_scores \
            --n 15 \
            --m 15 \
            --r 7 \
            --gamma 2.0 \
            --seed 0 \
            --wandb_name logging \
            --device 1 \
            --shuffle false \
            --score_calculator.class_path mixup.ood_score_calculators.$method \
            --score_calculator.init_args.batch_size 12288 \
            --datamodule.class_path mixup.ood_datamodules.$dataset
    done
done