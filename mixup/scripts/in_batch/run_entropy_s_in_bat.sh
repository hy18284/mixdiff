#!/bin/bash

for intermediate_state in softmax
do
    for ref_mode in in_batch
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
                for gamma in 2.0
                do
                    for n in 15
                    do
                        for p in 9
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
                                        --n $n \
                                        --m $m \
                                        --r $r \
                                        --p $p \
                                        --gamma $gamma \
                                        --r_ref 0 \
                                        --seed 0 \
                                        --wandb_name '' \
                                        --wandb_project ZOC \
                                        --wandb_tags aaai_sm \
                                        --device 0 \
                                        --ref_mode $ref_mode \
                                        --model_path 'ViT-B/32' \
                                        --max_samples null \
                                        --score_calculator.class_path mixup.ood_score_calculators.$method \
                                        --score_calculator.init_args.backbone.class_path mixup.ood_score_calculators.backbones.ClipBackbone \
                                        --score_calculator.init_args.batch_size 5000 \
                                        --score_calculator.init_args.log_interval null \
                                        --score_calculator.init_args.intermediate_state $intermediate_state \
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
    done
done