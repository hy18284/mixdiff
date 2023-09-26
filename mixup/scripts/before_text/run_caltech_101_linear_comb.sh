#!/bin/bash

batch_size=128

for base_method in \
    MixDiffMaxSofmaxProb \
    MixDiffEntropy \
    MixDiffEnergy \
    MixDiffMaxLogitScore 
do 
    for aux_method in \
        MixDiffMaxSofmaxProb \
        MixDiffEntropy \
        MixDiffEnergy \
        MixDiffMaxLogitScore 
    do 
        for dataset in \
            Caltech101OODDataset
        do
            for gamma_base in 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0
            do
                for gamma_exp in -4 -3 -2 -1 0 1 2 3 4
                do
                    gamma="${gamma_base}e${gamma_exp}"

                    if [ "$base_method" = "$aux_method" ]; then
                        continue
                    fi

                    python -m mixup.mixup_eval \
                        --n $batch_size \
                        --m 1 \
                        --r 1 \
                        --seed 0 \
                        --wandb_project ZOC \
                        --wandb_name baseline \
                        --device 0 \
                        --gamma $gamma \
                        --score_calculator.class_path mixup.ood_score_calculators.LinearCombination \
                        --score_calculator.init_args.base_ood_score_fn.class_path mixup.ood_score_calculators.$base_method \
                        --score_calculator.init_args.base_ood_score_fn.init_args.batch_size $batch_size \
                        --score_calculator.init_args.base_ood_score_fn.init_args.utilize_mixup false \
                        --score_calculator.init_args.base_ood_score_fn.init_args.add_base_score true \
                        --score_calculator.init_args.aux_ood_score_fn.class_path mixup.ood_score_calculators.$aux_method \
                        --score_calculator.init_args.aux_ood_score_fn.init_args.batch_size $batch_size \
                        --score_calculator.init_args.aux_ood_score_fn.init_args.utilize_mixup false \
                        --score_calculator.init_args.aux_ood_score_fn.init_args.add_base_score true \
                        --score_calculator.init_args.gamma  $gamma \
                        --datamodule.class_path mixup.ood_datamodules.$dataset
                done
            done
        done
    done
done