#!/bin/bash

for method in \
    MixDiffEntropy
do 
    for dataset in \
        CIFAR100OODDataset 
    do
        python -m mixup.save_mixup_scores \
            --n 15 \
            --m 15 \
            --r 7 \
            --gamma 2.0 \
            --seed 0 \
            --wandb_name logging \
            --device 6 \
            --score_calculator.class_path mixup.ood_score_calculators.$method \
            --score_calculator.init_args.batch_size 12288 \
            --datamodule.class_path mixup.ood_datamodules.$dataset \
            --datamodule.init_args.shuffle false
    done
done
        # CIFARPlus50OODDataset \ 
        # TinyImageNetOODDataset 
        # CIFAR10OODDataset \
        # CIFAR100OODDataset \
        # CIFARPlus10OODDataset \