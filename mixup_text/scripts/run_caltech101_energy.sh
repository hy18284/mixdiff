#!/bin/bash

for n in 15 10
do
    for m in 15 10
    do
        for r in 7 5
        do
            for gamma in 1.0 0.5 2.0
            do
                for method in \
                    MixDiffEnergy 
                do 
                    python -m mixup.mixup_eval \
                        --n $n \
                        --m $m \
                        --r $r \
                        --seed 0 \
                        --wandb_name paper_v1 \
                        --device 0 \
                        --gamma $gamma \
                        --score_calculator.class_path mixup.ood_score_calculators.$method \
                        --score_calculator.init_args.batch_size 1024 \
                        --score_calculator.init_args.utilize_mixup true \
                        --datamodule.class_path mixup.ood_datamodules.Caltech101OODDataset 
                done
            done
        done
    done
done
                    # MixDiffEntropy
                    # MixDiffMaxSofmaxProb 
                    # MixDiffEnergy 
                    # MixDiffMaxLogitScore 