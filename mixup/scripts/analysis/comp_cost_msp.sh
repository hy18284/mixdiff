#!/bin/bash

for intermediate_state in softmax
do
    for ref_mode in in_batch
    do
        for method in \
            MixDiffMaxSofmaxProb
        do 
            for dataset in \
                CIFAR100OODDataset
            do
                for r_n_g in \
                    "7 15 2" \
                    "7 12 2" \
                    "7 9 2" \
                    "7 6 2" \
                    "7 3 2" \
                    "7 2 1" \
                    "5 15 2" \
                    "5 12 2" \
                    "5 9 2" \
                    "5 6 2" \
                    "5 3 2" \
                    "5 2 1" \
                    "3 15 2" \
                    "3 12 2" \
                    "3 9 2" \
                    "3 6 2" \
                    "3 3 2" \
                    "3 2 0.5" \
                    "1 15 1" \
                    "1 12 1" \
                    "1 9 1" \
                    "1 6 0.5" \
                    "1 3 0.5" \
                    "1 2 0.5"
                do
                    for p in 3
                    do
                        for m in 15
                        do
                            if [ "$method" = "MixDiffEnergy" ] || [ "$method" = "MixDiffMaxLogitScore" ] && [ "$intermediate_state" = "softmax" ]; then
                                echo "Skipping ${method} ${intermediate_state}"
                                continue
                            fi
                            set -- $r_n_g
                            python -m mixup.mixup_eval_text \
                                --n $2 \
                                --m $m \
                                --r $1 \
                                --p $p \
                                --gamma $3 \
                                --r_ref 0 \
                                --seed 0 \
                                --wandb_name 'comp' \
                                --wandb_project ZOC_debug \
                                --wandb_tags comp_analysis \
                                --device 0 \
                                --ref_mode $ref_mode \
                                --model_path 'ViT-B/32' \
                                --max_samples null \
                                --score_calculator.class_path mixup.ood_score_calculators.$method \
                                --score_calculator.init_args.backbone.class_path mixup.ood_score_calculators.backbones.ClipBackbone \
                                --score_calculator.init_args.batch_size 5000 \
                                --score_calculator.init_args.log_interval null \
                                --score_calculator.init_args.intermediate_state $intermediate_state \
                                --score_calculator.init_args.backbone.init_args.post_transform false \
                                --score_calculator.init_args.oracle_sim_mode uniform \
                                --score_calculator.init_args.oracle_sim_temp 1.0 \
                                --score_calculator.init_args.utilize_mixup true \
                                --score_calculator.init_args.selection_mode argmax \
                                --score_calculator.init_args.add_base_score true \
                                --fnr_at 0.95 \
                                --fpr_at 0.95 \
                                --log_interval null \
                                --datamodule.class_path mixup.ood_datamodules.$dataset \
                                --mixup_operator.class_path mixup.mixup_operators.InterpolationMixup
                        done
                    done
                done
            done
        done
    done
done
                # CIFAR10OODDataset \
                # CIFARPlus10OODDataset \
                # CIFARPlus50OODDataset \
                # TinyImageNetOODDataset 
                # Caltech101OODDataset