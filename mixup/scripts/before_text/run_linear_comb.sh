#!/bin/bash

batch_size=128

for base_method in \
    MixDiffEntropy
do 
    for aux_method in \
        MixDiffEnergy
    do 
        for dataset in \
            CIFAR10OODDataset \
            CIFARPlus10OODDataset \
            CIFARPlus50OODDataset \
            CIFAR100OODDataset \
            TinyImageNetOODDataset 
        do
            for gamma_base in 1.0
            do
                for gamma_exp in 1
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

for base_method in \
    MixDiffEntropy
do 
    for aux_method in \
        MixDiffMaxSofmaxProb
    do 
        for dataset in \
            CIFAR100OODDataset
        do
            for gamma_base in 1.0
            do
                for gamma_exp in -4
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