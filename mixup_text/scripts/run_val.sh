#!/bin/bash

for method in \
    MixDiffEntropyText \
    MixDiffMaxSoftmaxProbText \
    MixDiffEnergyText \
    MixDiffMaxLogitScoreText 
do 
    for dataset in \
        acid \
        banking77 \
        clinic150 \
        snips \
        top 
    do
        for ref_pos in "both 2" "front 1" "rear 1"
        do
            for id_rate in 75 50 25
            do
                for gamma in 2.0 1.0 0.5
                do
                    for m in 30 25 20 15 10 5
                    do
                        set -- $ref_pos
                        python -m mixup_text.mixup_eval_text \
                            --n 258 \
                            --m $m \
                            --r $2 \
                            --gamma $gamma \
                            --r_ref 0 \
                            --seed 0 \
                            --wandb_name '' \
                            --wandb_project mixdiff \
                            --device 0 \
                            --ref_mode oracle \
                            --model_path bert-base-uncased \
                            --score_calculator.class_path mixup_text.ood_score_calculators.$method \
                            --score_calculator.init_args.batch_size 1000 \
                            --score_calculator.init_args.selection_mode argmax \
                            --fnr_at 0.95 \
                            --fpr_at 0.95 \
                            --datamodule.class_path mixup_text.ood_datamodules.ClassSplitOODDataset \
                            --datamodule.init_args.config_path mixup_text/configs_text/${dataset}_cs_val_$id_rate.yml \
                            --mixup_operator.class_path mixup_text.mixup_operators.ConcatMixup \
                            --mixup_operator.init_args.ref_pos $1
                    done
                done
            done
        done
    done
done