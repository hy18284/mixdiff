#!/bin/bash

for method in \
    MixDiffEnergyText \
    MixDiffMaxLogitScoreText \
    MixDiffEntropyText \
    MixDiffMaxSoftmaxProbText
do 
    for dataset in \
        acid
    do
        for ref_pos in "both 2"
        do
            for id_rate in 25 50 75
            do
                for gamma in 0.0
                do
                    for m in 1
                    do
                        set -- $ref_pos
                        python -m mixup.mixup_eval_text \
                            --n 258 \
                            --m $m \
                            --r $2 \
                            --gamma $gamma \
                            --r_ref 0 \
                            --seed 0 \
                            --wandb_name '' \
                            --wandb_project ZOC \
                            --device 0 \
                            --ref_mode oracle \
                            --model_path checkpoints/${dataset}_bert \
                            --score_calculator.class_path mixup.ood_score_calculators.$method \
                            --score_calculator.init_args.batch_size 10000 \
                            --score_calculator.init_args.selection_mode argmax \
                            --score_calculator.init_args.utilize_mixup false \
                            --score_calculator.init_args.add_base_score true \
                            --fnr_at 0.95 \
                            --fpr_at 0.95 \
                            --datamodule.class_path mixup.ood_datamodules.ClassSplitOODDataset \
                            --datamodule.init_args.config_path mixup/configs_text/${dataset}_cs_val_$id_rate.yml \
                            --mixup_operator.class_path mixup.mixup_operators.ConcatMixup \
                            --mixup_operator.init_args.ref_pos $1
                    done
                done
            done
        done
    done
done
                            # --score_calculator.init_args.utilize_mixup false \
                            # --score_calculator.init_args.add_base_score true \