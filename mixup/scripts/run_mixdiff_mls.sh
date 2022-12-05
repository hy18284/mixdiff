#!/bin/bash

for method in \
    MixDiffMaxLogitScore
do 
    for dataset in \
        CIFAR10OODDataset \
        CIFAR100OODDataset \
        CIFARPlus10OODDataset \
        CIFARPlus50OODDataset \
        TinyImageNetOODDataset 
    do
        python -m mixup.mixup_eval \
            --config mixup/configs/mixdiff_msp_config.yaml \
            --n 15 \
            --m 10 \
            --r 7 \
            --seed 0 \
            --wandb_name paper_v1 \
            --device 7 \
            --gamma 1.0 \
            --score_calculator.class_path mixup.ood_score_calculators.$method \
            --score_calculator.init_args.batch_size 12288 \
            --datamodule.class_path mixup.ood_datamodules.$dataset
    done
done