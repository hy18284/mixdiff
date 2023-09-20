#!/bin/bash

for method in \
    MixDiffMaxSofmaxProb
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
            --device 2 \
            --score_calculator.class_path mixup.ood_score_calculators.$method \
            --score_calculator.init_args.batch_size 12288 \
            --datamodule.class_path mixup.ood_datamodules.$dataset \
            --datamodule.init_args.shuffle false
    done
done
        # CIFAR100OODDataset \
        # TinyImageNetOODDataset 
        # CIFAR10OODDataset \
        # CIFARPlus10OODDataset 