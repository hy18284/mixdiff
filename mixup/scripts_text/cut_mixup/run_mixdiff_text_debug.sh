#!/bin/bash

for method in \
    MixDiffEntropyText
do 
    for dataset in \
        "CLINIC150OODDataset clinic150"
    do
        for mixup_fn in \
            CutMixup
        do
            for n in 5
            do
                for m in 5
                do
                    for r in 3
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
                                    --model_path "checkpoints/${2}_bert" \
                                    --score_calculator.class_path mixup.ood_score_calculators.$method \
                                    --score_calculator.init_args.batch_size 20000 \
                                    --score_calculator.init_args.selection_mode $selection_mode \
                                    --datamodule.class_path mixup.ood_datamodules.$1 \
                                    --datamodule.init_args.mode val \
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

for method in \
    MixDiffEntropyText
do 
    for dataset in \
        "CLINIC150OODDataset clinic150"
    do
        for mixup_fn in \
            CutMixup
        do
            for n in 5
            do
                for m in 5
                do
                    for r in 3
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
                                    --model_path "checkpoints/${2}_bert" \
                                    --score_calculator.class_path mixup.ood_score_calculators.$method \
                                    --score_calculator.init_args.batch_size 20000 \
                                    --score_calculator.init_args.selection_mode $selection_mode \
                                    --datamodule.class_path mixup.ood_datamodules.$1 \
                                    --datamodule.init_args.mode val \
                                    --mixup_operator.class_path mixup.mixup_operators.$mixup_fn \
                                    --mixup_operator.init_args.model_path roberta-base
                            done
                        done
                    done
                done
            done
        done
    done
done

for method in \
    MixDiffEntropyText
do 
    for dataset in \
        "CLINIC150OODDataset clinic150"
    do
        for mixup_fn in \
            CutMixup
        do
            for n in 5
            do
                for m in 5
                do
                    for r in 3
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
                                    --model_path "checkpoints/${2}_bert" \
                                    --score_calculator.class_path mixup.ood_score_calculators.$method \
                                    --score_calculator.init_args.batch_size 20000 \
                                    --score_calculator.init_args.selection_mode $selection_mode \
                                    --datamodule.class_path mixup.ood_datamodules.$1 \
                                    --datamodule.init_args.mode val \
                                    --mixup_operator.class_path mixup.mixup_operators.$mixup_fn \
                                    --mixup_operator.init_args.model_path checkpoints/clinic150_bert
                            done
                        done
                    done
                done
            done
        done
    done
done

for method in \
    MixDiffEntropyText
do 
    for dataset in \
        "CLINIC150OODDataset clinic150"
    do
        for mixup_fn in \
            CutMixup
        do
            for n in 20
            do
                for m in 20
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
                                    --model_path "checkpoints/${2}_bert" \
                                    --score_calculator.class_path mixup.ood_score_calculators.$method \
                                    --score_calculator.init_args.batch_size 20000 \
                                    --score_calculator.init_args.selection_mode $selection_mode \
                                    --datamodule.class_path mixup.ood_datamodules.$1 \
                                    --datamodule.init_args.mode val \
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

for method in \
    MixDiffEntropyText
do 
    for dataset in \
        "CLINIC150OODDataset clinic150"
    do
        for mixup_fn in \
            CutMixup
        do
            for n in 20
            do
                for m in 20
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
                                    --model_path "checkpoints/${2}_bert" \
                                    --score_calculator.class_path mixup.ood_score_calculators.$method \
                                    --score_calculator.init_args.batch_size 20000 \
                                    --score_calculator.init_args.selection_mode $selection_mode \
                                    --datamodule.class_path mixup.ood_datamodules.$1 \
                                    --datamodule.init_args.mode val \
                                    --mixup_operator.class_path mixup.mixup_operators.$mixup_fn \
                                    --mixup_operator.init_args.model_path roberta-base
                            done
                        done
                    done
                done
            done
        done
    done
done

for method in \
    MixDiffEntropyText
do 
    for dataset in \
        "CLINIC150OODDataset clinic150"
    do
        for mixup_fn in \
            CutMixup
        do
            for n in 20
            do
                for m in 20
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
                                    --model_path "checkpoints/${2}_bert" \
                                    --score_calculator.class_path mixup.ood_score_calculators.$method \
                                    --score_calculator.init_args.batch_size 20000 \
                                    --score_calculator.init_args.selection_mode $selection_mode \
                                    --datamodule.class_path mixup.ood_datamodules.$1 \
                                    --datamodule.init_args.mode val \
                                    --mixup_operator.class_path mixup.mixup_operators.$mixup_fn \
                                    --mixup_operator.init_args.model_path checkpoints/clinic150_bert
                            done
                        done
                    done
                done
            done
        done
    done
done