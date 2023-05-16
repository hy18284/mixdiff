#!/bin/bash

for method in \
    MixDiffMaxLogitScoreText
do 
    for dataset in \
        "TopOODDataset top" \
        "Banking77OODDatasetClinicTest banking77" \
        "AcidOODDatasetClinicTest acid" \
        "SnipsOODDatasetClinicTest snips" \
        "AcidOODDatasetClinicWiki acid" \
        "Banking77OODDatasetClinicWiki banking77" \
        "SnipsOODDatasetClinicWiki snips"
    do
        for mixup_fn in \
            CutMixup
        do
            for m in 5
            do
                for gamma in 0.5
                do
                    for n in 258
                    do
                        for r in 5
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
                                    --inner_seeds 0 1 2 3 4 \
                                    --wandb_name cln_tst_pr \
                                    --wandb_project ZOC \
                                    --device 0 \
                                    --ref_mode 'oracle' \
                                    --model_path "checkpoints/${2}_bert" \
                                    --score_calculator.class_path mixup.ood_score_calculators.$method \
                                    --score_calculator.init_args.batch_size 10000 \
                                    --score_calculator.init_args.selection_mode $selection_mode \
                                    --datamodule.class_path mixup.ood_datamodules.$1 \
                                    --datamodule.init_args.mode test \
                                    --mixup_operator.class_path mixup.mixup_operators.$mixup_fn \
                                    --mixup_operator.init_args.model_path bert-base-uncased
                            done
                        done
                    done
                done
            done
        done
    done
done
        # "CLINIC150OODDataset clinic150" \
        # "CLINIC150OODDatasetWiki clinic150" \