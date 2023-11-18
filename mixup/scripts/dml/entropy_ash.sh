#!/bin/bash

for intermediate_state in softmax
do
    for ref_mode in in_batch
    do
        for method in \
            MixDiffEntropy
        do 
            for dataset in \
                CIFARPlus50OODDataset 
            do
                for gamma in 2.0
                do
                    for n in 100
                    do
                        for p in 3
                        do
                            for m in 15
                            do
                                for r in 7
                                do
                                for ash_p in  10 20 30 40
                                do
                                for ash_method in ash_b ash_p ash_s
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
                                            --wandb_name "${ash_method}@${ash_p}" \
                                            --wandb_project ZOC_debug \
                                            --wandb_tags ash \
                                            --device 0 \
                                            --ref_mode $ref_mode \
                                            --model_path 'ViT-B/32' \
                                            --max_samples null \
                                            --trans_before_mixup true \
                                            --score_calculator.class_path mixup.ood_score_calculators.$method \
                                            --score_calculator.init_args.backbone.class_path mixup.ood_score_calculators.backbones.ClipBackboneASH \
                                            --score_calculator.init_args.backbone.init_args.ash_method "${ash_method}@${ash_p}" \
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
                # Caltech101OODDataset
                                                # --datamodule.init_args.val_ratio 0.66 \
                                        # for attack in id2ood ood2id both none
                # CIFAR10OODDataset
                # CIFARPlus50OODDataset \
                # TinyImageNetOODDataset
                # for attack in none id2ood ood2id both