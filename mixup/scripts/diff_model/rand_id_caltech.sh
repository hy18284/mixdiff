#!/bin/bash
for ref_mode in rand_id
do
    for method in \
        MixDiffMaxSofmaxProb \
        MixDiffEntropy
    do 
        for dataset in \
            Caltech101OODDataset
        do
            for backbone in \
                RN50 \
                ViT-L/14 \
                RN50x64
            do
                for gamma in 2.0 1.0 0.5
                do
                    for m in 15 10
                    do
                        for p in 14 9
                        do
                            for r in 7 5
                            do
                                if [ "$method" = "MixDiffEnergy" ] || [ "$method" = "MixDiffMaxLogitScore" ] && [ "$intermediate_state" = "softmax" ]; then
                                    echo "Skipping ${method} ${intermediate_state}"
                                    continue
                                fi
                                python -m mixup.mixup_eval_text \
                                    --n 10 \
                                    --m $m \
                                    --r $r \
                                    --p $p \
                                    --gamma $gamma \
                                    --r_ref 0 \
                                    --seed 0 \
                                    --wandb_name $backbone \
                                    --wandb_project ZOC_dev \
                                    --wandb_tags diff_model_v1 \
                                    --device 0 \
                                    --model_path $backbone \
                                    --max_samples null \
                                    --trans_before_mixup true \
                                    --ref_mode $ref_mode \
                                    --score_calculator.class_path mixup.ood_score_calculators.$method \
                                    --score_calculator.init_args.backbone.class_path mixup.ood_score_calculators.backbones.ClipBackbone \
                                    --score_calculator.init_args.batch_size 100 \
                                    --score_calculator.init_args.log_interval null \
                                    --score_calculator.init_args.intermediate_state softmax \
                                    --fnr_at 0.95 \
                                    --fpr_at 0.95 \
                                    --log_interval null \
                                    --datamodule.class_path mixup.ood_datamodules.$dataset \
                                    --datamodule.init_args.val_ratio 0.66 \
                                    --datamodule.init_args.with_replacement true  \
                                    --mixup_operator.class_path mixup.mixup_operators.InterpolationMixup
                            done
                        done
                    done
                done
            done
        done
    done
done