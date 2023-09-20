#!/bin/bash

for method in \
    MixDiffMaxLogitScoreText
do 
    for dataset in \
        ClassSplitOODDataset
    do
        for mixup_fn in \
            SplitMixup
        do
            for id_rate in 75 50 25
            do
                for m in 20 15 10 5
                do
                    for gamma in 1.0 0.5 2.0
                    do
                        for n in 258
                        do
                            for r in 11 9 7 5
                            do
                                python -m mixup.mixup_eval_text \
                                    --n $n \
                                    --m $m \
                                    --r $r \
                                    --gamma $gamma \
                                    --r_ref 0 \
                                    --seed 0 \
                                    --wandb_name '' \
                                    --wandb_project ZOC \
                                    --device 0 \
                                    --ref_mode oracle \
                                    --model_path checkpoints/clinic150_bert \
                                    --score_calculator.class_path mixup.ood_score_calculators.$method \
                                    --score_calculator.init_args.batch_size 20000 \
                                    --score_calculator.init_args.selection_mode argmax \
                                    --fnr_at 0.95 \
                                    --fpr_at 0.95 \
                                    --datamodule.class_path mixup.ood_datamodules.$dataset \
                                    --datamodule.init_args.config_path mixup/configs_text/clinic150_cs_val_$id_rate.yml \
                                    --mixup_operator.class_path mixup.mixup_operators.$mixup_fn
                            done
                        done
                    done
                done
            done
        done
    done
done