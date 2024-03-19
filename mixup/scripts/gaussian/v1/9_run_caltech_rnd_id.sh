#!/bin/bash

for intermediate_state in softmax
do
    for ref_mode in in_batch
    do
        for method in \
            MixDiffMaxSofmaxProb \
            MixDiffEntropy
        do 
            for dataset in \
                Caltech101OODDataset
            do
                for gamma in 2.0 1.0 0.5
                do
                    for n in 15 10
                    do
                        for m in 15 10
                        do
                            for vars in "0.5, 0.25, 0.1, 0.05, 0.025, 0.01, 0.005, 0.0025, 0.001"
                            do
                                if [ "$method" = "MixDiffEnergy" ] || [ "$method" = "MixDiffMaxLogitScore" ] && [ "$intermediate_state" = "softmax" ]; then
                                    echo "Skipping ${method} ${intermediate_state}"
                                    continue
                                fi
                                python -m mixup.mixup_eval_text \
                                    --n $n \
                                    --m $m \
                                    --r 1 \
                                    --p 9 \
                                    --gamma $gamma \
                                    --r_ref 0 \
                                    --seed 0 \
                                    --wandb_name '' \
                                    --wandb_project ZOC \
                                    --wandb_tags gaussian_v1 \
                                    --wandb_tags \
                                    --device 0 \
                                    --ref_mode $ref_mode \
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
                                    --mixup_operator.init_args.vars "[$vars]"
                            done
                        done
                    done
                done
            done
        done
    done
done
                            # for vars in [0.1] [0.01] [0.001] [0.1, 0.01, 0.001] [0.1, 0.01] [0.01, 0.001] [0.1, 0.001]