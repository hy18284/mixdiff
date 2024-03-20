#!/bin/bash

for intermediate_state in softmax
do
    for method in \
        MixDiffEntropy
    do 
        for dataset in \
            CIFAR10OODDataset \
            CIFARPlus10OODDataset \
            CIFARPlus50OODDataset \
            CIFAR100OODDataset \
            TinyImageNetOODDataset 
        do
            for gamma in 0.5
            do
                for m in 15
                do
                    for p in 10
                    do
                        for vars in "0.05, 0.025, 0.01, 0.005, 0.001"
                        do
                            if [ "$method" = "MixDiffEnergy" ] || [ "$method" = "MixDiffMaxLogitScore" ] && [ "$intermediate_state" = "softmax" ]; then
                                echo "Skipping ${method} ${intermediate_state}"
                                continue
                            fi
                            python -m mixup.mixup_eval_text \
                                --n 100 \
                                --m $m \
                                --r 5 \
                                --p $p \
                                --gamma $gamma \
                                --r_ref 0 \
                                --seed 0 \
                                --wandb_name '' \
                                --wandb_project ZOC \
                                --wandb_tags gaussian_v6 \
                                --device 0 \
                                --model_path 'ViT-B/32' \
                                --max_samples null \
                                --trans_before_mixup false \
                                --ref_mode 'rand_id' \
                                --score_calculator.class_path mixup.ood_score_calculators.$method \
                                --score_calculator.init_args.backbone.class_path mixup.ood_score_calculators.backbones.ClipBackbone \
                                --score_calculator.init_args.batch_size 10000 \
                                --score_calculator.init_args.log_interval null \
                                --score_calculator.init_args.intermediate_state $intermediate_state \
                                --fnr_at 0.95 \
                                --fpr_at 0.95 \
                                --log_interval null \
                                --datamodule.class_path mixup.ood_datamodules.$dataset \
                                --mixup_operator.class_path mixup.mixup_operators.GaussianNoise \
                                --mixup_operator.init_args.stds "[$vars]"
                        done
                    done
                done
            done
        done
    done
done