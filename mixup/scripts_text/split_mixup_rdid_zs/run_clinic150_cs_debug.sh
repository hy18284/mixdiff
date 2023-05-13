#!/bin/bash

for method in \
    MixDiffMaxLogitScoreTextZS
do 
    for dataset in \
        ClassSplitOODDataset
    do
        for mixup_fn in \
            SplitMixup
        do
            for gamma in 1.0
            do
                for n in 512
                do
                    for m in 3
                    do
                        for r in 3
                        do
                            for id_ratio in 25 50 75
                            do
                                python -m mixup.mixup_eval_text \
                                    --n $n \
                                    --m $m \
                                    --r $r \
                                    --gamma $gamma \
                                    --r_ref 0 \
                                    --seed 0 \
                                    --wandb_name debug \
                                    --wandb_project ZOC_debug \
                                    --device 0 \
                                    --ref_mode 'rand_id' \
                                    --model_path bert-base-uncased \
                                    --score_calculator.class_path mixup.ood_score_calculators.$method \
                                    --score_calculator.init_args.batch_size 20000 \
                                    --score_calculator.init_args.utilize_mixup false \
                                    --score_calculator.init_args.add_base_score true \
                                    --datamodule.class_path mixup.ood_datamodules.$dataset \
                                    --datamodule.init_args.config_path mixup/configs_text/clinic150_cs_val_$id_ratio.yml \
                                    --mixup_operator.class_path mixup.mixup_operators.$mixup_fn
                            done
                        done
                    done
                done
            done
        done
    done
done
                                    # --score_calculator.init_args.utilize_mixup false \
                                    # --score_calculator.init_args.add_base_score true \