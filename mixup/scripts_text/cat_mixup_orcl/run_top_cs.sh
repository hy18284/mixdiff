#!/bin/bash

for method in \
    MixDiffMaxLogitScoreText
do 
    for dataset in \
        ClassSplitOODDataset
    do
        for mixup_fn in \
            ConcatMixup
        do
            for ref_pos in "both 2" "front 1" "rear 1"
            do
                for id_rate in 75 50 25
                do
                    for gamma in 1.0 0.5 2.0
                    do
                        for m in 30 25 20 10 5 1
                        do
                            set -- $ref_pos
                            python -m mixup.mixup_eval_text \
                                --n 258 \
                                --m $m \
                                --r $2 \
                                --gamma $gamma \
                                --r_ref 0 \
                                --seed 0 \
                                --wandb_name debug \
                                --wandb_project ZOC_debug \
                                --device 0 \
                                --ref_mode oracle \
                                --model_path checkpoints/clinic150_bert \
                                --score_calculator.class_path mixup.ood_score_calculators.$method \
                                --score_calculator.init_args.batch_size 20000 \
                                --score_calculator.init_args.selection_mode argmax \
                                --fnr_at 0.95 \
                                --fpr_at 0.95 \
                                --datamodule.class_path mixup.ood_datamodules.$dataset \
                                --datamodule.init_args.config_path mixup/configs_text/top_cs_val_$id_rate.yml \
                                --mixup_operator.class_path mixup.mixup_operators.$mixup_fn \
                                --mixup_operator.init_args.ref_pos $1
                        done
                    done
                done
            done
        done
    done
done