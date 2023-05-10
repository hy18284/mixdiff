#!/bin/bash

for method in \
    MixDiffEnergyText
do 
    for dataset in \
        "CLINIC150OODDataset clinic150" \
        "CLINIC150OODDatasetWiki clinic150" \
        "TopOODDataset top" \
        "Banking77OODDatasetClinicTest banking77" \
        "Banking77OODDatasetClinicWiki banking77" \
        "AcidOODDatasetClinicTest acid" \
        "AcidOODDatasetClinicWiki acid" \
        "SnipsOODDatasetClinicTest snips" \
        "SnipsOODDatasetClinicWiki snips"
    do
        for mixup_fn in \
            SplitMixup
        do
            for n in 10
            do
                for m in 20
                do
                    for r in 7
                    do
                        for gamma in 2
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
                                    --wandb_name cln_tst \
                                    --wandb_project ZOC \
                                    --device 0 \
                                    --ref_mode 'oracle' \
                                    --model_path "checkpoints/${2}_bert" \
                                    --score_calculator.class_path mixup.ood_score_calculators.$method \
                                    --score_calculator.init_args.batch_size 1000000000 \
                                    --score_calculator.init_args.selection_mode $selection_mode \
                                    --datamodule.class_path mixup.ood_datamodules.$1 \
                                    --datamodule.init_args.mode test \
                                    --mixup_operator.class_path mixup.mixup_operators.$mixup_fn
                            done
                        done
                    done
                done
            done
        done
    done
done