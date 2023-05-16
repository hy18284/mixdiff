#!/bin/bash

for method in \
    MixDiffMaxLogitScoreTextZS
do 
    for dataset in \
        "CLINIC150OODDataset clinic150" \
        "CLINIC150OODDatasetWiki clinic150" \
        "TopOODDataset top" \
        "Banking77OODDatasetClinicTest banking77" \
        "AcidOODDatasetClinicTest acid" \
        "SnipsOODDatasetClinicTest snips" \
        "AcidOODDatasetClinicWiki acid" \
        "Banking77OODDatasetClinicWiki banking77" \
        "SnipsOODDatasetClinicWiki snips"
    do
        for mixup_fn in \
            ConcatMixup
        do
            for n in 258
            do
                for m in 20
                do
                    for r in 7
                    do
                        for gamma in 2.0
                        do
                            for selection_mode in argmax
                            do
                                set -- $dataset
                                python -m mixup.mixup_eval_text \
                                    --n $n \
                                    --m $m \
                                    --r 2 \
                                    --gamma $gamma \
                                    --r_ref 0 \
                                    --seed 0 \
                                    --wandb_name cln_tst \
                                    --wandb_project ZOC_debug \
                                    --device 0 \
                                    --ref_mode 'oracle' \
                                    --model_path "bert-base-uncased" \
                                    --score_calculator.class_path mixup.ood_score_calculators.$method \
                                    --score_calculator.init_args.utilize_mixup false \
                                    --score_calculator.init_args.add_base_score true \
                                    --score_calculator.init_args.batch_size 100 \
                                    --score_calculator.init_args.selection_mode $selection_mode \
                                    --datamodule.class_path mixup.ood_datamodules.$1 \
                                    --datamodule.init_args.mode test \
                                    --mixup_operator.class_path mixup.mixup_operators.$mixup_fn \
                                    --mixup_operator.init_args.ref_pos both
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