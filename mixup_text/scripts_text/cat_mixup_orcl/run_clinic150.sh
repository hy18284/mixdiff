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
            CutMixup
        do
            for m in 20 15 10 5
            do
                for gamma in 1.0 0.5 2.0
                do
                    for n in 258
                    do
                        for r in 11 9 7 5
                        do
                            for selection_mode in argmax euclidean dot
                            do
                                python -m mixup.mixup_eval_text \
                                    --n $n \
                                    --m $m \
                                    --r $r \
                                    --gamma $gamma \
                                    --r_ref 0 \
                                    --seed 0 \
                                    --wandb_name cln_val \
                                    --wandb_project ZOC \
                                    --device 0 \
                                    --ref_mode 'oracle' \
                                    --model_path checkpoints/clinic150_bert \
                                    --score_calculator.class_path mixup.ood_score_calculators.$method \
                                    --score_calculator.init_args.batch_size 10000000 \
                                    --score_calculator.init_args.selection_mode $selection_mode \
                                    --datamodule.class_path mixup.ood_datamodules.$dataset \
                                    --datamodule.init_args.mode val \
                                    --mixup_operator.class_path mixup.mixup_operators.$mixup_fn
                            done
                        done
                    done
                done
            done
        done
    done
done