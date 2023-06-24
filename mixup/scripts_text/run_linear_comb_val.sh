#!/bin/bash

batch_size=128

for base_method in \
    MixDiffMaxSoftmaxProbText \
    MixDiffEntropyText \
    MixDiffEnergyText \
    MixDiffMaxLogitScoreText
do 
    for aux_method in \
        MixDiffMaxSoftmaxProbText \
        MixDiffEntropyText \
        MixDiffEnergyText \
        MixDiffMaxLogitScoreText
    do 
        for id_rate in 75 50 25
        do
            for gamma_base in 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0
            do
                for gamma_exp in -4 -3 -2 -1 0 1 2 3 4
                do
                    gamma="${gamma_base}e${gamma_exp}"

                    if [ "$base_method" = "$aux_method" ]; then
                        continue
                    fi

                    python -m mixup.mixup_eval_text \
                        --n $batch_size \
                        --m 1 \
                        --r 1 \
                        --r_ref 0 \
                        --seed 0 \
                        --wandb_project ZOC \
                        --wandb_name baseline \
                        --device 0 \
                        --gamma $gamma \
                        --ref_mode rand_id \
                        --model_path checkpoints/clinic150_bert \
                        --fnr_at 0.95 \
                        --fpr_at 0.95 \
                        --score_calculator.class_path mixup.ood_score_calculators.LinearCombinationText \
                        --score_calculator.init_args.base_ood_score_fn.class_path mixup.ood_score_calculators.$base_method \
                        --score_calculator.init_args.base_ood_score_fn.init_args.batch_size $batch_size \
                        --score_calculator.init_args.base_ood_score_fn.init_args.selection_mode argmax \
                        --score_calculator.init_args.base_ood_score_fn.init_args.utilize_mixup false \
                        --score_calculator.init_args.base_ood_score_fn.init_args.add_base_score true \
                        --score_calculator.init_args.aux_ood_score_fn.class_path mixup.ood_score_calculators.$aux_method \
                        --score_calculator.init_args.aux_ood_score_fn.init_args.batch_size $batch_size \
                        --score_calculator.init_args.aux_ood_score_fn.init_args.selection_mode argmax \
                        --score_calculator.init_args.aux_ood_score_fn.init_args.utilize_mixup false \
                        --score_calculator.init_args.aux_ood_score_fn.init_args.add_base_score true \
                        --score_calculator.init_args.gamma  $gamma \
                        --datamodule.class_path mixup.ood_datamodules.$dataset \
                        --datamodule.init_args.config_path mixup/configs_text/clinic150_cs_val_$id_rate.yml \
                        --mixup_operator.class_path mixup.mixup_operators.ConcatMixup \
                        --mixup_operator.init_args.ref_pos both
                done
            done
        done
    done
done