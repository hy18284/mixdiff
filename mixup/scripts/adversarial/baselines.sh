#!/bin/bash

for intermediate_state in softmax
do
    for ref_mode in in_batch
    do
        for method in \
            MixDiffEntropy \
            MixDiffMaxSofmaxProb 
        do 
            for dataset in \
                CIFAR100OODDataset \
                CIFAR10OODDataset
            do
                for eps in 1
                do
                for attack_nb_iter in 10 1 5 20 50 100
                do
                    for attack in both id2ood ood2id
                    do
                        for gamma in 1.0
                        do
                            for n in 1000
                            do
                                for p in 3
                                do
                                    for m in 15
                                    do
                                        for r in 3
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
                                                --wandb_name "adv_$attack" \
                                                --wandb_project ZOC \
                                                --wandb_tags adv \
                                                --device 0 \
                                                --ref_mode $ref_mode \
                                                --model_path 'ViT-B/32' \
                                                --max_samples null \
                                                --attack $attack \
                                                --attack_eps $eps \
                                                --attack_nb_iter $attack_nb_iter \
                                                --trans_before_mixup true \
                                                --score_calculator.class_path mixup.ood_score_calculators.$method \
                                                --score_calculator.init_args.backbone.class_path mixup.ood_score_calculators.backbones.ClipBackbone \
                                                --score_calculator.init_args.batch_size 5000 \
                                                --score_calculator.init_args.log_interval null \
                                                --score_calculator.init_args.intermediate_state $intermediate_state \
                                                --score_calculator.init_args.oracle_sim_mode uniform \
                                                --score_calculator.init_args.oracle_sim_temp 1.0 \
                                                --score_calculator.init_args.utilize_mixup false \
                                                --score_calculator.init_args.selection_mode argmax \
                                                --score_calculator.init_args.add_base_score true \
                                                --fnr_at 0.95 \
                                                --fpr_at 0.95 \
                                                --log_interval null \
                                                --datamodule.class_path mixup.ood_datamodules.$dataset \
                                                --datamodule.init_args.with_replacement true \
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
done
done