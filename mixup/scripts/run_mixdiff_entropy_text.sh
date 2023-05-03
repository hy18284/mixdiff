#!/bin/bash

for method in \
    MixDiffEntropyText
do 
    for dataset in \
        CLINIC150OODDataset
    do
        for mixup_fn in \
            EmbeddingMixup
        do
            for n in 15 10
            do
                for m in 15 10
                do
                    for r in 7 5
                    do
                        for gamma in 1.0 0.5 2.0
                        do
                            python -m mixup.mixup_eval_text \
                                --n $n \
                                --m $m \
                                --r $r \
                                --gamma $gamma \
                                --seed 0 \
                                --wandb_name hp \
                                --device 0 \
                                --model_path checkpoints/clinic150_bert \
                                --score_calculator.class_path mixup.ood_score_calculators.$method \
                                --score_calculator.init_args.batch_size 4096 \
                                --datamodule.class_path mixup.ood_datamodules.$dataset \
                                --datamodule.init_args.mode val \
                                --mixup_operator.class_path mixup.mixup_operators.$mixup_fn \
                                --mixup_operator.init_args.model_path roberta-base \
                                --mixup_operator.init_args.device 0 \
                                --mixup_operator.init_args.interpolation truncate \
                                --mixup_operator.init_args.similarity dot
                        done
                    done
                done
            done
        done
    done
done