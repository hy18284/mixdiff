#!/bin/bash

count=0

for method in \
    MixDiffMaxLogitScoreText
do 
    for dataset in \
        ClassSplitOODDataset
    do
        for mixup_fn in \
            ConcatMixup
        do
            for ref_pos in "both 2"
            do
                for id_rate in 75
                do
                    for gamma in 2.0
                    do
                        for m in 25
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
                                --model_path checkpoints/clinic150_bert \
                                --score_calculator.class_path mixup.ood_score_calculators.$method \
                                --score_calculator.init_args.batch_size 5000 \
                                --score_calculator.init_args.selection_mode argmax \
                                --fnr_at 0.95 \
                                --fpr_at 0.95 \
                                --datamodule.class_path mixup.ood_datamodules.$dataset \
                                --datamodule.init_args.config_path mixup/configs_text/clinic150_cs_test_$id_rate.yml \
                                --mixup_operator.class_path mixup.mixup_operators.$mixup_fn \
                                --mixup_operator.init_args.ref_pos $1
                        done
                    done
                done
            done
        done
    done
done


for method in \
    MixDiffMaxLogitScoreText
do 
    for dataset in \
        ClassSplitOODDataset
    do
        for mixup_fn in \
            ConcatMixup
        do
            for ref_pos in "both 2"
            do
                for id_rate in 50
                do
                    for gamma in 2.0
                    do
                        for m in 25
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
                                --model_path checkpoints/clinic150_bert \
                                --score_calculator.class_path mixup.ood_score_calculators.$method \
                                --score_calculator.init_args.batch_size 5000 \
                                --score_calculator.init_args.selection_mode argmax \
                                --fnr_at 0.95 \
                                --fpr_at 0.95 \
                                --datamodule.class_path mixup.ood_datamodules.$dataset \
                                --datamodule.init_args.config_path mixup/configs_text/clinic150_cs_test_$id_rate.yml \
                                --mixup_operator.class_path mixup.mixup_operators.$mixup_fn \
                                --mixup_operator.init_args.ref_pos $1
                        done
                    done
                done
            done
        done
    done
done


for method in \
    MixDiffMaxLogitScoreText
do 
    for dataset in \
        ClassSplitOODDataset
    do
        for mixup_fn in \
            ConcatMixup
        do
            for ref_pos in "both 2"
            do
                for id_rate in 25
                do
                    for gamma in 2.0
                    do
                        for m in 5
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
                                --model_path checkpoints/clinic150_bert \
                                --score_calculator.class_path mixup.ood_score_calculators.$method \
                                --score_calculator.init_args.batch_size 5000 \
                                --score_calculator.init_args.selection_mode argmax \
                                --fnr_at 0.95 \
                                --fpr_at 0.95 \
                                --datamodule.class_path mixup.ood_datamodules.$dataset \
                                --datamodule.init_args.config_path mixup/configs_text/clinic150_cs_test_$id_rate.yml \
                                --mixup_operator.class_path mixup.mixup_operators.$mixup_fn \
                                --mixup_operator.init_args.ref_pos $1
                        done
                    done
                done
            done
        done
    done
done
                            # ((count=count+1))

                            # if (( count < 4 )); then
                            #     continue
                            #     echo "skipping $count"
                            # fi