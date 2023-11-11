for method in \
    MixDiffMaxSofmaxProb
do 
    for dataset in \
        "data/Places places" \
        "data/iNaturalist inat" \
        "data/SUN sun" \
        "data/dtd/images textures"
    do
        for gamma in 2.0
        do
            for n in 100
            do
                for m in 2
                do
                    for p in 2
                    do
                        for r in 2
                        do
                            for sim_temp in 0.01
                            do
                                for selection_mode in argmax
                                do
                                    set -- $dataset
                                    python -m mixup.mixup_eval_text \
                                        --n $n \
                                        --m $m \
                                        --r $r \
                                        --p $p \
                                        --gamma $gamma \
                                        --r_ref 0 \
                                        --seed 0 \
                                        --wandb_name '' \
                                        --wandb_tags \
                                        --wandb_project ZOC \
                                        --device 0 \
                                        --ref_mode in_batch \
                                        --model_path 'ViT-B/32' \
                                        --max_samples null \
                                        --id_as_neg false \
                                        --score_calculator.class_path mixup.ood_score_calculators.$method \
                                        --score_calculator.init_args.backbone.class_path mixup.ood_score_calculators.backbones.ClipBackbone \
                                        --score_calculator.init_args.backbone.init_args.post_transform true \
                                        --score_calculator.init_args.batch_size 2000 \
                                        --score_calculator.init_args.log_interval null \
                                        --score_calculator.init_args.intermediate_state softmax \
                                        --score_calculator.init_args.oracle_sim_mode uniform \
                                        --score_calculator.init_args.oracle_sim_temp $sim_temp \
                                        --score_calculator.init_args.utilize_mixup false \
                                        --score_calculator.init_args.selection_mode $selection_mode \
                                        --score_calculator.init_args.add_base_score true \
                                        --fnr_at 0.95 \
                                        --fpr_at 0.95 \
                                        --log_interval null \
                                        --datamodule.class_path mixup.ood_datamodules.ImageNetCrossOODDataset \
                                        --datamodule.init_args.id_dataset_dir data/imagenet/val \
                                        --datamodule.init_args.ood_dataset_dir $1 \
                                        --datamodule.init_args.name imgnet_$2 \
                                        --datamodule.init_args.oracle_data_dir data/imagenet/train/ \
                                        --mixup_operator.class_path mixup.mixup_operators.InterpolationMixup
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done