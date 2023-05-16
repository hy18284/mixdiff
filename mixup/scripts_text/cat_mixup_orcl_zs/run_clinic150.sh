#!/bin/bash

for method in \
    MixDiffMaxLogitScoreTextZS
do 
    for dataset in \
        CLINIC150OODDataset
    do
        for mixup_fn in \
            ConcatMixup
        do
            for ref_pos in "both 2" "front 1" "rear 1"
            do
                for gamma in 1.0
                do
                    for n in 258
                    do
                        for m in 3
                        do
                            for r in 3
                            do
                                set -- $ref_pos
                                python -m mixup.mixup_eval_text \
                                    --n $n \
                                    --m $m \
                                    --r $2 \
                                    --gamma $gamma \
                                    --r_ref 0 \
                                    --seed 0 \
                                    --wandb_name '' \
                                    --wandb_project ZOC \
                                    --device 0 \
                                    --ref_mode 'oracle' \
                                    --model_path bert-base-uncased \
                                    --score_calculator.class_path mixup.ood_score_calculators.$method \
                                    --score_calculator.init_args.batch_size 10000 \
                                    --datamodule.class_path mixup.ood_datamodules.$dataset \
                                    --datamodule.init_args.mode val \
                                    --mixup_operator.class_path mixup.mixup_operators.$mixup_fn \
                                    --mixup_operator.init_args.ref_pos $1
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
                                        # --score_calculator.init_args.utilize_mixup false \
                                        # --score_calculator.init_args.add_base_score true \