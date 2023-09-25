#!/bin/bash

for method in \
    MixCosSoftmaxProb 
do 
    for dataset in \
        ImageNetCrossOODDataset
    do
        for mixup_fn in \
            InterpolationMixup
        do
            for gamma in 2.0 1.0 0.5
            do
                for n in 1
                do
                    for m in 7
                    do
                        for p in 1
                        do
                            for r in 5
                            do
                                for sim_temp in 0.01
                                do
                                    for selection_mode in argmax
                                    do
                                        python -m mixup.mixup_eval_text \
                                            --n $n \
                                            --m $m \
                                            --r $r \
                                            --p $p \
                                            --gamma $gamma \
                                            --r_ref 0 \
                                            --seed 0 \
                                            --wandb_name clip_post \
                                            --wandb_project ZOC_debug \
                                            --device 0 \
                                            --ref_mode oracle \
                                            --model_path 'ViT-B/32' \
                                            --max_samples null \
                                            --id_as_neg false \
                                            --score_calculator.class_path mixup.ood_score_calculators.$method \
                                            --score_calculator.init_args.backbone.class_path mixup.ood_score_calculators.backbones.ClipBackbone \
                                            --score_calculator.init_args.backbone.init_args.post_transform true \
                                            --score_calculator.init_args.batch_size 1 \
                                            --score_calculator.init_args.log_interval null \
                                            --score_calculator.init_args.intermediate_state softmax \
                                            --score_calculator.init_args.oracle_sim_mode uniform \
                                            --score_calculator.init_args.oracle_sim_temp $sim_temp \
                                            --score_calculator.init_args.utilize_mixup true \
                                            --score_calculator.init_args.selection_mode $selection_mode \
                                            --score_calculator.init_args.add_base_score true \
                                            --fnr_at 0.95 \
                                            --fpr_at 0.95 \
                                            --log_interval null \
                                            --datamodule.class_path mixup.ood_datamodules.$dataset \
                                            --datamodule.init_args.ood_dataset_dir data/dtd/images \
                                            --datamodule.init_args.name imgnet_textures \
                                            --mixup_operator.class_path mixup.mixup_operators.$mixup_fn
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

        # CIFAR10CrossOODDataset