#!/bin/bash

for n in 10 15
do
    for m in 10 15
    do
        for r in 5 7
        do
            for gamma in 0.5 1.0 2.0
            do
                for method in \
                    MixDiffZOC
                do 
                    for dataset in \
                        CIFAR10OODDataset \
                        CIFAR100OODDataset \
                        CIFARPlus10OODDataset \
                        CIFARPlus50OODDataset \
                        TinyImageNetOODDataset \
                        Caltech101OODDataset
                    do
                        python -m mixup.mixup_eval \
                            --n $n \
                            --m $m \
                            --r $r \
                            --gamma $gamma \
                            --seed 0 \
                            --wandb_name plan_b_hp_partial \
                            --device 0 \
                            --max_samples 200 \
                            --score_calculator.class_path mixup.ood_score_calculators.$method \
                            --score_calculator.init_args.batch_size 150 \
                            --score_calculator.init_args.zoc_checkpoint_path trained_models/COCO/ViT-B32/model.pt \
                            --score_calculator.init_args.utilize_mixup true \
                            --score_calculator.init_args.add_base_scores true \
                            --score_calculator.init_args.follow_zoc true \
                            --score_calculator.init_args.half_precision true \
                            --score_calculator.init_args.avg_logits exp \
                            --datamodule.class_path mixup.ood_datamodules.$dataset
                    done
                done
            done
        done
    done
done