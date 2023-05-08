#!/bin/bash

for method in \
    MixDiffEntropyText \
    MixDiffMaxSofmaxProbText \
    MixDiffEnergyText \
    MixDiffMaxLogitScoreText 
do 
    for dataset in \
        "Banking77OODDatasetClinicTest banking77" \
        "Banking77OODDatasetClinicWiki banking77" \
        "AcidOODDatasetClinicTest acid" \
        "AcidOODDatasetClinicWiki acid" \
        "SnipsOODDatasetClinicTest snips" \
        "SnipsOODDatasetClinicWiki snips" \
        "CLINIC150OODDataset clinic150" \
        "CLINIC150OODDatasetWiki clinic150" \
        "TopOODDataset top" 
    do
        for mixup_fn in \
            StringMixup
        do
            for n in 1000
            do
                for m in 1
                do
                    for r in 1
                    do
                        for gamma in 0.0
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
                                    --wandb_project ZOC_debug \
                                    --device 0 \
                                    --model_path "checkpoints/${2}_bert" \
                                    --score_calculator.class_path mixup.ood_score_calculators.$method \
                                    --score_calculator.init_args.batch_size 1000000 \
                                    --score_calculator.init_args.selection_mode $selection_mode \
                                    --score_calculator.init_args.utilize_mixup false \
                                    --score_calculator.init_args.add_base_score true \
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