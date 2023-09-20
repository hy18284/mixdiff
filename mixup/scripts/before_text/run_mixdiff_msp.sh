#!/bin/bash

for method in \
    MixDiffMaxSofmaxProb
do 
    for dataset in \
        CIFAR10OODDataset \
        CIFAR100OODDataset \
        CIFARPlus10OODDataset \
        CIFARPlus50OODDataset \
        TinyImageNetOODDataset 
    do
        python -m mixup.mixup_eval \
            --n 15 \
            --m 15 \
            --r 7 \
            --gamma 2.0 \
            --seed 0 \
            --wandb_name val_caltech \
            --device 0 \
            --score_calculator.class_path mixup.ood_score_calculators.$method \
            --score_calculator.init_args.batch_size 12288 \
            --datamodule.class_path mixup.ood_datamodules.$dataset
    done
done