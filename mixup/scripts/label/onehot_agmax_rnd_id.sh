#!/bin/bash

for intermediate_state in one_hot
do
    for ref_mode in rand_id
    do
        for method in \
            MixDiffOneHot
        do 
            for dataset in \
                CIFAR10OODDataset \
                CIFARPlus10OODDataset \
                CIFARPlus50OODDataset \
                CIFAR100OODDataset \
                TinyImageNetOODDataset 
            do
                for selection_mode in argmax
                do
                    for gamma in 1.0
                    do
                        for n in 3
                        do
                            for p in 14
                            do
                                for m in 15
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
                                            --wandb_project mixdiff \
                                            --wandb_tags onehot_v1 \
                                            --device 0 \
                                            --ref_mode $ref_mode \
                                            --model_path 'ViT-B/32' \
                                            --max_samples null \
                                            --score_calculator.class_path mixup.ood_score_calculators.$method \
                                            --score_calculator.init_args.backbone.class_path mixup.ood_score_calculators.backbones.ClipBackbone \
                                            --score_calculator.init_args.batch_size 100 \
                                            --score_calculator.init_args.log_interval null \
                                            --score_calculator.init_args.intermediate_state $intermediate_state \
                                            --score_calculator.init_args.selection_mode $selection_mode \
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
done

