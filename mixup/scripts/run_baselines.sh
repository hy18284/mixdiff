#!/bin/bash

    # MixDiffMaxSofmaxProb \
    # MixDiffEnergy \
    # MixDiffMaxLogitScore 

        # CIFAR10OODDataset \
        # CIFARPlus10OODDataset \
        # CIFARPlus50OODDataset \
for method in \
    MixDiffEntropy 
do 
    for dataset in \
        CIFAR100OODDataset \
        TinyImageNetOODDataset 
    do
        python -m mixup.mixup_eval \
            --config mixup/configs/mixdiff_msp_config.yaml \
            --n 3 \
            --m 15 \
            --r 7 \
            --seed 0 \
            --wandb_name paper_v1 \
            --device 2 \
            --gamma 1.0 \
            --score_calculator.class_path mixup.ood_score_calculators.$method \
            --score_calculator.init_args.batch_size 2048 \
            --score_calculator.init_args.utilize_mixup false \
            --datamodule.class_path mixup.ood_datamodules.$dataset
    done
done