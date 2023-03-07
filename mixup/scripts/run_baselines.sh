#!/bin/bash

for method in \
    MixDiffEntropy \
    MixDiffMaxSofmaxProb \
    MixDiffEnergy \
    MixDiffMaxLogitScore 
do 
    for dataset in \
        Caltech101OODDataset \
        CIFAR10OODDataset \
        CIFARPlus10OODDataset \
        CIFARPlus50OODDataset \
        CIFAR100OODDataset \
        TinyImageNetOODDataset 
    do
        python -m mixup.mixup_eval \
            --n 15 \
            --m 15 \
            --r 7 \
            --seed 0 \
            --wandb_name baseline \
            --device 7 \
            --gamma 1.0 \
            --score_calculator.class_path mixup.ood_score_calculators.$method \
            --score_calculator.init_args.batch_size 2048 \
            --score_calculator.init_args.utilize_mixup false \
            --score_calculator.init_args.add_base_score true \
            --datamodule.class_path mixup.ood_datamodules.$dataset
    done
done