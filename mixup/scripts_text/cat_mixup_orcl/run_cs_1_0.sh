#!/bin/bash

    # MixDiffEnergyText \
    # MixDiffEntropyText \
    # MixDiffMaxLogitScoreText \
    # MixDiffMaxSoftmaxProbText

# for ref_pos in "both 2" "front 1" "rear 1"
for method in \
    MixDiffEnergyText
do 
    for dataset in \
        banking77 \
        acid \
        top
    do
        for ref_pos in "front 1" "rear 1"
        do
            for id_rate in 75 50 25
            do
                for gamma in 1.0 2.0 0.5
                do
                    for m in 30 25 20 15 10 5
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
                            --wandb_tags cls_fixed \
                            --device 0 \
                            --ref_mode oracle \
                            --model_path checkpoints/${dataset}_bert \
                            --score_calculator.class_path mixup.ood_score_calculators.$method \
                            --score_calculator.init_args.batch_size 5000 \
                            --score_calculator.init_args.selection_mode argmax \
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
            # for id_rate in 25 50 75
            # do
            #     for gamma in 1.0 0.5 2.0
            #     do
            #         for m in 30 25 20 15 10 5
            #         do