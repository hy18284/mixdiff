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
            python -m mixup.mixup_eval_text \
                --n 3 \
                --m 3 \
                --r 3 \
                --gamma 2.0 \
                --seed 0 \
                --wandb_name cln_test \
                --device 0 \
                --model_path checkpoints/clinic150_bert \
                --score_calculator.class_path mixup.ood_score_calculators.$method \
                --score_calculator.init_args.batch_size 258 \
                --score_calculator.init_args.utilize_mixup false \
                --score_calculator.init_args.add_base_score true \
                --datamodule.class_path mixup.ood_datamodules.$dataset \
                --datamodule.init_args.mode test \
                --mixup_operator.class_path mixup.mixup_operators.$mixup_fn \
                --mixup_operator.init_args.model_path roberta-base \
                --mixup_operator.init_args.device 0 \
                --mixup_operator.init_args.interpolation pad \
                --mixup_operator.init_args.similarity dot
        done
    done
done