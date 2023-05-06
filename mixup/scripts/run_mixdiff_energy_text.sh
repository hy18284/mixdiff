#!/bin/bash

for method in \
    MixDiffEnergyText \
    MixDiffEntropyText \
    MixDiffMaxLogitScoreText \
    MixDiffMaxSofmaxProbText
do 
    for dataset in \
        CLINIC150OODDataset
    do
        for mixup_fn in \
            EmbeddingMixup
        do
            for gamma in 1.0
            do
                for n in 15
                do
                    for m in 15
                    do
                        for r in 7
                        do
                            for selection_mode in argmax
                            do
                                python -m mixup.mixup_eval_text \
                                    --n $n \
                                    --m $m \
                                    --r $r \
                                    --gamma $gamma \
                                    --r_ref 0.25 \
                                    --seed 0 \
                                    --wandb_name cln_val \
                                    --wandb_project ZOC \
                                    --device 0 \
                                    --model_path checkpoints/clinic150_bert \
                                    --score_calculator.class_path mixup.ood_score_calculators.$method \
                                    --score_calculator.init_args.batch_size 4096 \
                                    --score_calculator.init_args.selection_mode $selection_mode \
                                    --datamodule.class_path mixup.ood_datamodules.$dataset \
                                    --datamodule.init_args.mode val \
                                    --mixup_operator.class_path mixup.mixup_operators.$mixup_fn \
                                    --mixup_operator.init_args.model_path roberta-base \
                                    --mixup_operator.init_args.device 0 \
                                    --mixup_operator.init_args.interpolation pad \
                                    --mixup_operator.init_args.similarity dot
                            done
                        done
                    done
                done
            done
        done
    done
done