#!/bin/bash
for ref_mode in in_batch
do
    for method in \
        MixDiffMaxSofmaxProb
    do 
        for dataset in \
            CIFAR10OODDataset \
            CIFARPlus10OODDataset \
            CIFARPlus50OODDataset \
            CIFAR100OODDataset \
            TinyImageNetOODDataset 
        do
            for backbone in \
                ViT-B/32
            do
                for gamma in 2.0
                do
                    for n in 15
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
                                    --p 9 \
                                    --gamma $gamma \
                                    --r_ref 0 \
                                    --seed 0 \
                                    --wandb_name $backbone \
                                    --wandb_project mixdiff \
                                    --wandb_tags prob_v1 \
                                    --device 0 \
                                    --model_path $backbone \
                                    --max_samples null \
                                    --trans_before_mixup true \
                                    --ref_mode $ref_mode \
                                    --score_calculator.class_path mixup.ood_score_calculators.$method \
                                    --score_calculator.init_args.backbone.class_path mixup.ood_score_calculators.backbones.ClipBackbone \
                                    --score_calculator.init_args.batch_size 1000 \
                                    --score_calculator.init_args.log_interval null \
                                    --score_calculator.init_args.intermediate_state softmax \
                                    --score_calculator.init_args.utilize_mixup true \
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