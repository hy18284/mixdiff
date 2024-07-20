#!/bin/bash

for n in 15 10
do
    for m in 15 10
    do
        for r in 7 5
        do
            for gamma in 2.0 1.0 0.5
            do
                for method in \
                    MixDiffZOC
                do 
                    python -m mixup.mixup_eval \
                        --n $n \
                        --m $m \
                        --r $r \
                        --gamma $gamma \
                        --seed 0 \
                        --wandb_name plan_b_partial \
                        --device 6 \
                        --max_samples 50 \
                        --score_calculator.class_path mixup.ood_score_calculators.$method \
                        --score_calculator.init_args.batch_size 160 \
                        --score_calculator.init_args.zoc_checkpoint_path trained_models/COCO/ViT-B32/model.pt \
                        --score_calculator.init_args.utilize_mixup true \
                        --score_calculator.init_args.add_base_scores true \
                        --score_calculator.init_args.follow_zoc true \
                        --datamodule.class_path mixup.ood_datamodules.Caltech101OODDataset 
                done
            done
        done
    done
done