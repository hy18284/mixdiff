#!/bin/bash
for ref_mode in oracle
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
            for backbone in \
                RN50x4
            do
                for gamma in 1.0
                do
                    for m in 10
                    do
                        for r in 7
                        do
                            if [ "$method" = "MixDiffEnergy" ] || [ "$method" = "MixDiffMaxLogitScore" ] && [ "$intermediate_state" = "softmax" ]; then
                                echo "Skipping ${method} ${intermediate_state}"
                                continue
                            fi
                            python -m mixup.mixup_eval_text \
                                --n 10 \
                                --m $m \
                                --r $r \
                                --p $m \
                                --gamma $gamma \
                                --r_ref 0 \
                                --seed 0 \
                                --wandb_name $backbone \
                                --wandb_project ZOC \
                                --wandb_tags diff_model_v1 \
                                --device 0 \
                                --model_path $backbone \
                                --max_samples null \
                                --trans_before_mixup true \
                                --ref_mode $ref_mode \
                                --score_calculator.class_path mixup.ood_score_calculators.$method \
                                --score_calculator.init_args.backbone.class_path mixup.ood_score_calculators.backbones.ClipBackbone \
                                --score_calculator.init_args.batch_size 500 \
                                --score_calculator.init_args.log_interval null \
                                --score_calculator.init_args.intermediate_state softmax \
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