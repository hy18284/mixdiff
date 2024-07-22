#!/bin/bash 

for method in \
    MixDiffEntropyText \
    MixDiffMaxSoftmaxProbText \
    MixDiffEnergyText \
    MixDiffMaxLogitScoreText 
do 
    # Choose the dataset from below.
    for dataset in \
        acid \
        banking77 \
        clinic150 \
        snips \
        top 
    do
        # Choose the in-scope class ratio.
        for id_rate in 75 50 25
        do
            python -m mixup_text.mixup_eval_text \
                --n 258 \
                --m 5 \
                --r 2 \
                --gamma 1.0 \
                --r_ref 0 \
                --seed 0 \
                --wandb_name '' \
                --wandb_project mixdiff \
                --device 0 \
                --ref_mode in_batch \
                --model_path bert-base-uncased \
                --score_calculator.class_path mixup_text.ood_score_calculators.$method \
                --score_calculator.init_args.batch_size 1000 \
                --score_calculator.init_args.selection_mode argmax \
                --score_calculator.init_args.utilize_mixup false \
                --fnr_at 0.95 \
                --fpr_at 0.95 \
                --datamodule.class_path mixup_text.ood_datamodules.ClassSplitOODDataset \
                --datamodule.init_args.config_path mixup_text/configs_text/${dataset}_cs_test_$id_rate.yml \
                --mixup_operator.class_path mixup_text.mixup_operators.ConcatMixup \
                --mixup_operator.init_args.ref_pos both
        done
    done
done