for method in \
    MixCosEntropy \
    MixDotEntropy \
    MixDiffEntropy \
    MixCosMaxSoftmaxProb \
    MixDotMaxSoftmaxProb \
    MixDiffMaxSofmaxProb
    
do 
    for dataset in \
        "data/Places places" \
        "data/iNaturalist inat" \
        "data/SUN sun" \
        "data/dtd/images textures"
    do
    do
        for gamma in 2.0 1.0 0.5
        do
            for n in 10
            do
                for m in 15 10
                do
                    for p in 14 9
                    do
                        for r in 7 5
                        do
                            for sim_temp in 0.01
                            do
                                for selection_mode in argmax
                                do
                                    python -m mixup.mixup_eval_text \
                                        --n $n \
                                        --m $m \
                                        --r $r \
                                        --p $p \
                                        --gamma $gamma \
                                        --r_ref 0 \
                                        --seed 0 \
                                        --wandb_name post \
                                        --wandb_tags post_transform \
                                        --wandb_project ZOC \
                                        --device 0 \
                                        --ref_mode rand_id \
                                        --model_path 'ViT-B/32' \
                                        --max_samples null \
                                        --id_as_neg true \
                                        --score_calculator.class_path mixup.ood_score_calculators.$method \
                                        --score_calculator.init_args.backbone.class_path mixup.ood_score_calculators.backbones.ClipBackbone \
                                        --score_calculator.init_args.backbone.init_args.post_transform true \
                                        --score_calculator.init_args.batch_size 100 \
                                        --score_calculator.init_args.log_interval null \
                                        --score_calculator.init_args.intermediate_state softmax \
                                        --score_calculator.init_args.oracle_sim_mode uniform \
                                        --score_calculator.init_args.oracle_sim_temp $sim_temp \
                                        --score_calculator.init_args.utilize_mixup true \
                                        --score_calculator.init_args.selection_mode $selection_mode \
                                        --score_calculator.init_args.add_base_score true \
                                        --fnr_at 0.95 \
                                        --fpr_at 0.95 \
                                        --log_interval null \
                                        --datamodule.class_path mixup.ood_datamodules.CrossDatasetOODDataset \
                                        --datamodule.init_args.id_dataset_dir data/imagenet/val \
                                        --datamodule.init_args.ood_dataset_dir $1 \
                                        --datamodule.init_args.name imgnet_$2 \
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