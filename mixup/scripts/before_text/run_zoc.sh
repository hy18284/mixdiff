#!/bin/bash

for n in 150
do
    for m in 15
    do
        for r in 7
        do
            for gamma in 2.0
            do
                for method in \
                    MixDiffZOC
                do 
                    for dataset in \
                        Caltech101OODDataset \
                        CIFARPlus50OODDataset \
                        CIFAR10OODDataset \
                        CIFAR100OODDataset \
                        CIFARPlus10OODDataset \
                        TinyImageNetOODDataset 
                    do
                        python -m mixup.mixup_eval \
                            --n $n \
                            --m $m \
                            --r $r \
                            --gamma $gamma \
                            --seed 0 \
                            --wandb_name plan_b_hp_partial \
                            --device 1 \
                            --max_samples 200 \
                            --score_calculator.class_path mixup.ood_score_calculators.$method \
                            --score_calculator.init_args.batch_size 150 \
                            --score_calculator.init_args.zoc_checkpoint_path trained_models/COCO/ViT-B32/model.pt \
                            --score_calculator.init_args.utilize_mixup false \
                            --score_calculator.init_args.add_base_scores true \
                            --score_calculator.init_args.follow_zoc true \
                            --score_calculator.init_args.half_precision true \
                            --score_calculator.init_args.avg_base_logits exp \
                            --datamodule.class_path mixup.ood_datamodules.$dataset
                    done
                done
            done
        done
    done
done