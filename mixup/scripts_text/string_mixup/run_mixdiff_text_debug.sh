#!/bin/bash

for method in \
    MixDiffEntropyText
do 
    for dataset in \
        "CLINIC150OODDataset clinic150"
    do
        for mixup_fn in \
            StringMixup
        do
            for n in 15
            do
                for m in 15
                do
                    for r in 7
                    do
                        for gamma in 1.0 
                        do
                            for selection_mode in argmax
                            do
                                set -- $dataset
                                python -m mixup.mixup_eval_text \
                                    --n $n \
                                    --m $m \
                                    --r $r \
                                    --gamma $gamma \
                                    --r_ref 0 \
                                    --seed 0 \
                                    --wandb_name cln_val \
                                    --wandb_project ZOC_debug \
                                    --device 0 \
                                    --max_samples 1000 \
                                    --model_path "checkpoints/${2}_bert" \
                                    --score_calculator.class_path mixup.ood_score_calculators.$method \
                                    --score_calculator.init_args.batch_size 10000 \
                                    --score_calculator.init_args.selection_mode $selection_mode \
                                    --datamodule.class_path mixup.ood_datamodules.$1 \
                                    --datamodule.init_args.mode val \
                                    --mixup_operator.class_path mixup.mixup_operators.$mixup_fn
                            done
                        done
                    done
                done
            done
        done
    done
done