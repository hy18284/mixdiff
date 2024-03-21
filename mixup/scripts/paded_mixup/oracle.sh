#!/bin/bash
for ref_mode in oracle
do
    for method in \
        MixDiffMaxSofmaxProb \
        MixDiffEntropy
    do 
        for ratio in 0.146 0.25
        do
            for dataset in \
                CIFAR10OODDataset \
                CIFAR100OODDataset \
                CIFARPlus50OODDataset \
                CIFARPlus10OODDataset \
                TinyImageNetOODDataset 
            do
                for gamma in 2.0
                do
                    for m in 15
                    do
                        for p in 9
                        do
                            for r in 7
                            do
                                if [ "$method" = "MixDiffEnergy" ] || [ "$method" = "MixDiffMaxLogitScore" ] && [ "$intermediate_state" = "softmax" ]; then
                                    echo "Skipping ${method} ${intermediate_state}"
                                    continue
                                fi
                                python -m mixup.mixup_eval_text \
                                    --n 100 \
                                    --m $m \
                                    --r $r \
                                    --p $p \
                                    --gamma $gamma \
                                    --r_ref 0 \
                                    --seed 0 \
                                    --wandb_name $ratio \
                                    --wandb_project ZOC \
                                    --wandb_tags paded_mixup_v2 \
                                    --device 0 \
                                    --model_path 'ViT-B/32' \
                                    --max_samples null \
                                    --trans_before_mixup true \
                                    --ref_mode $ref_mode \
                                    --score_calculator.class_path mixup.ood_score_calculators.$method \
                                    --score_calculator.init_args.backbone.class_path mixup.ood_score_calculators.backbones.ClipBackbone \
                                    --score_calculator.init_args.batch_size 10000 \
                                    --score_calculator.init_args.log_interval null \
                                    --score_calculator.init_args.intermediate_state softmax \
                                    --fnr_at 0.95 \
                                    --fpr_at 0.95 \
                                    --log_interval null \
                                    --datamodule.class_path mixup.ood_datamodules.$dataset \
                                    --mixup_operator.class_path mixup.mixup_operators.PadedInterpolationMixup \
                                    --mixup_operator.init_args.ratio $ratio
                            done
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
                # CIFAR100OODDataset \
                # TinyImageNetOODDataset 