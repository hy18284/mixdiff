#!/bin/bash

for method in \
    MixDiffOODClass 
do 
    for dataset in \
        top 
    do
        for ref_pos in "rear 1"
        do
            for id_rate in 50
            do
                for gamma in 1.0
                do
                    for m in 2
                    do
                        set -- $ref_pos
                        python -m mixup.mixup_eval_text \
                            --n 5 \
                            --m $m \
                            --r $2 \
                            --k 50 \
                            --gamma $gamma \
                            --r_ref 0 \
                            --seed 0 \
                            --wandb_name openai_debug \
                            --wandb_project ZOC_debug \
                            --model_path gpt-3.5-turbo \
                            --device 0 \
                            --ref_mode oracle \
                            --id_as_neg true \
                            --max_samples 100 \
                            --score_calculator.class_path mixup.ood_score_calculators.$method \
                            --score_calculator.init_args.backbone.class_path mixup.ood_score_calculators.backbones.OpenAIBackbone \
                            --score_calculator.init_args.backbone.init_args.n_examples 2 \
                            --score_calculator.init_args.backbone.init_args.n_calls 3 \
                            --score_calculator.init_args.batch_size 10000 \
                            --score_calculator.init_args.selection_mode argmax \
                            --score_calculator.init_args.intermediate_state logit \
                            --score_calculator.init_args.oracle_sim_mode uniform \
                            --score_calculator.init_args.utilize_mixup true \
                            --score_calculator.init_args.add_base_score true \
                            --fnr_at 0.95 \
                            --fpr_at 0.95 \
                            --datamodule.class_path mixup.ood_datamodules.ClassSplitOODDataset \
                            --datamodule.init_args.config_path mixup/configs_text/reduced/${dataset}_cs_test_$id_rate.yml \
                            --mixup_operator.class_path mixup.mixup_operators.ConcatMixup \
                            --mixup_operator.init_args.ref_pos $1
                    done
                done
            done
        done
    done
done

    # for dataset in \
    #     acid \
    #     banking77 \
    #     clinic150 \
    #     snips \
    #     top 
    # do
                            # --score_calculator.init_args.log_interval null \