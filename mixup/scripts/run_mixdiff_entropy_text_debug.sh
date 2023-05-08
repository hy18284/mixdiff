#!/bin/bash

for method in \
    MixDiffEntropyText
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
            StringMixup
        do
            for n in 2
            do
                for m in 2
                do
                    for r in 2
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
                                    --wandb_name cln_tst \
                                    --wandb_project ZOC_debug \
                                    --device 0 \
                                    --max_samples 200 \
                                    --model_path "checkpoints/${2}_bert" \
                                    --score_calculator.class_path mixup.ood_score_calculators.$method \
                                    --score_calculator.init_args.batch_size 258 \
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
                                    # --mixup_operator.init_args.model_path roberta-base \
                                    # --mixup_operator.init_args.device 0 \
                                    # --mixup_operator.init_args.interpolation pad \
                                    # --mixup_operator.init_args.similarity dot

                                    # --score_calculator.init_args.utilize_mixup false \
                                    # --score_calculator.init_args.add_base_score true \

        # CLINIC150OODDataset \
        # "AcidOODDataset acid"
                                    # --max_samples 300 \